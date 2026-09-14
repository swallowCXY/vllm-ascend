#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Business lifecycle metadata manager for KV cache control.

KVCacheControlManager (KVCM) holds request-level lifecycle declarations and
exposes the agent-facing control interfaces (pin / set_ttl / release). It owns
metadata only: no KV tensors, no block objects. Effects on the block pool are
applied by callers (patches / engine control methods) through the read-only
query APIs (``protection_level``, release plans). See:
docs/source/developer_guide/Design_Documents/KV_Cache_Control_Manager.md
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vllm_ascend import envs

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = getLogger(__name__)

_KV_CACHE_CONTROL_KEY = "kv_cache_control"
_MODE_NO_STORE = "no_store"
_MODE_PIN = "pin"
_MODE_TTL = "ttl"
_PARSED_ATTR = "_kv_cache_control_parsed"
_UNPARSED = object()
_STATE_PENDING = "pending"
_STATE_ACTIVE = "active"


@dataclass(frozen=True)
class ContentRef:
    """Agent-facing reference to cacheable content.

    ``cache_key`` must be declared by at least one request via
    ``kv_cache_control`` so the manager can resolve it to block hashes.
    ``prefix_tokens`` optionally limits the reference to a token-count prefix
    (floored to the engine block size at binding time).
    """

    namespace: str = "default"
    cache_key: str | None = None
    request_id: str | None = None
    prefix_tokens: int | None = None


class PinHandle(str):
    """Opaque handle returned by ``pin`` for later release."""

    @classmethod
    def generate(cls) -> PinHandle:
        return cls(uuid4().hex)


@dataclass
class LifecycleEntry:
    """Lifecycle metadata for one (namespace, cache_key)."""

    namespace: str
    cache_key: str
    policy: str
    priority: int
    pin_count: int
    ttl_s: float | None
    expire_at: float | None
    hashes: frozenset | None
    state: str


class KVCacheControlManager:
    """Holds lifecycle declarations and answers scheduling-time queries.

    One instance is attached to ``KVCacheManager`` (scheduler side, bound to
    the block pool for quota accounting) and one to the AscendStore connector
    scheduler (external delete queueing). Hot-path queries are O(1) and
    memoized per request.
    """

    def __init__(self) -> None:
        self.metrics: dict[str, int] = {
            "no_store_requests": 0,
            "unsupported_requests": 0,
            "parse_errors": 0,
            "pin_requests": 0,
            "ttl_requests": 0,
            "quota_degraded": 0,
            "ttl_expired": 0,
            "release_calls": 0,
            "pin_evictions": 0,
        }
        self._entries: dict[tuple[str, str], LifecycleEntry] = {}
        self._hash_index: dict[Any, set[tuple[str, str]]] = {}
        self._handle_index: dict[PinHandle, tuple[str, str]] = {}
        self._next_expiry: float | None = None
        self._kv_cache_manager: Any = None
        self._block_size: int | None = None
        self._release_plan: list[Any] = []

    def bind_kv_cache_manager(self, kv_cache_manager: Any, block_size: int | None = None) -> None:
        """Bind the owning KVCacheManager for quota accounting and eviction."""
        self._kv_cache_manager = kv_cache_manager
        self._block_size = block_size

    # ------------------------------------------------------------------
    # Request-level declaration parsing
    # ------------------------------------------------------------------

    def parse_request_control(self, request: Request) -> dict[str, Any] | None:
        """Parse and memoize the request-level ``kv_cache_control`` declaration.

        Returns the normalized control dict, or ``None`` when the request
        carries no (supported) declaration. The result is memoized on the
        request object so repeated scheduling-step calls are O(1).
        """
        cached = getattr(request, _PARSED_ATTR, _UNPARSED)
        if cached is not _UNPARSED:
            return cached
        control = self._parse_control(getattr(request, "kv_transfer_params", None))
        setattr(request, _PARSED_ATTR, control)
        return control

    def is_no_store(self, request: Request) -> bool:
        """Whether the request declared ``mode: no_store``."""
        control = self.parse_request_control(request)
        return control is not None and control["mode"] == _MODE_NO_STORE

    def _parse_control(self, kv_transfer_params: Any) -> dict[str, Any] | None:
        if kv_transfer_params is None:
            return None
        if not isinstance(kv_transfer_params, dict):
            self.metrics["parse_errors"] += 1
            logger.warning("Invalid kv_transfer_params type, ignoring: %r", kv_transfer_params)
            return None
        raw = kv_transfer_params.get(_KV_CACHE_CONTROL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict) or not isinstance(raw.get("mode"), str):
            self.metrics["parse_errors"] += 1
            logger.warning("Invalid kv_cache_control payload, ignoring: %r", raw)
            return None
        mode = raw["mode"]
        if mode == _MODE_NO_STORE:
            self.metrics["no_store_requests"] += 1
            return {"mode": _MODE_NO_STORE}
        if mode in (_MODE_PIN, _MODE_TTL):
            cache_key = raw.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                self.metrics["parse_errors"] += 1
                logger.warning("kv_cache_control mode %r requires a string cache_key", mode)
                return None
            control: dict[str, Any] = {
                "mode": mode,
                "cache_key": cache_key,
                "priority": raw.get("priority", 0) if isinstance(raw.get("priority", 0), int) else 0,
                "tier": raw.get("tier", "hbm") if isinstance(raw.get("tier", "hbm"), str) else "hbm",
                "ttl_s": raw.get("ttl_s") if isinstance(raw.get("ttl_s"), (int, float)) else None,
                "prefix_tokens": raw.get("prefix_tokens") if isinstance(raw.get("prefix_tokens"), int) else None,
            }
            if mode == _MODE_TTL and not control["ttl_s"]:
                self.metrics["parse_errors"] += 1
                logger.warning("kv_cache_control mode ttl requires a positive ttl_s")
                return None
            if mode == _MODE_PIN:
                self.metrics["pin_requests"] += 1
            else:
                self.metrics["ttl_requests"] += 1
            return control
        self.metrics["unsupported_requests"] += 1
        logger.debug("Unknown kv_cache_control mode %r, ignoring", mode)
        return None

    # ------------------------------------------------------------------
    # Lifecycle registration
    # ------------------------------------------------------------------

    def on_request_finished(self, request: Request) -> None:
        """Capture block hashes at request finish and activate declarations.

        Called from the ``KVCacheManager.free`` wrapper before the request's
        blocks are returned to the free queue.
        """
        control = self.parse_request_control(request)
        if control is None or control["mode"] == _MODE_NO_STORE:
            return
        hashes = self._capture_hashes(request, control.get("prefix_tokens"))
        self._upsert_entry(
            namespace="default",
            cache_key=control["cache_key"],
            policy=control["mode"],
            priority=control.get("priority", 0),
            ttl_s=control.get("ttl_s"),
            hashes=hashes,
        )

    def _capture_hashes(self, request: Request, prefix_tokens: int | None) -> list[Any]:
        raw = list(getattr(request, "block_hashes", None) or [])
        if prefix_tokens is not None and self._block_size:
            keep = max(0, int(prefix_tokens) // self._block_size)
            raw = raw[:keep]
        return raw

    def _upsert_entry(
        self,
        *,
        namespace: str,
        cache_key: str,
        policy: str,
        priority: int,
        ttl_s: float | None,
        hashes: list[Any] | None,
    ) -> LifecycleEntry | None:
        key = (namespace, cache_key)
        entry = self._entries.get(key)
        if hashes is not None and len(hashes) == 0:
            hashes = None
        if hashes is None:
            if entry is None:
                entry = LifecycleEntry(
                    namespace=namespace,
                    cache_key=cache_key,
                    policy=policy,
                    priority=priority,
                    pin_count=1,
                    ttl_s=ttl_s,
                    expire_at=time.monotonic() + ttl_s if ttl_s else None,
                    hashes=None,
                    state=_STATE_PENDING,
                )
                self._entries[key] = entry
                self._recompute_next_expiry()
            return entry
        if not self._quota_allows(len(hashes)):
            self.metrics["quota_degraded"] += 1
            logger.warning(
                "kv_cache_control pin/ttl for key %r exceeds pin budget, degrading to normal",
                cache_key,
            )
            if entry is not None and entry.state == _STATE_PENDING:
                self._unregister(key)
            return None
        if entry is not None and entry.hashes is not None:
            self._unindex_hashes(key, entry.hashes)
        entry = LifecycleEntry(
            namespace=namespace,
            cache_key=cache_key,
            policy=policy,
            priority=priority,
            pin_count=max(entry.pin_count, 1) if entry is not None else 1,
            ttl_s=ttl_s,
            expire_at=time.monotonic() + ttl_s if ttl_s else None,
            hashes=frozenset(hashes),
            state=_STATE_ACTIVE,
        )
        self._entries[key] = entry
        self._index_hashes(key, entry.hashes)
        self._recompute_next_expiry()
        return entry

    # ------------------------------------------------------------------
    # Agent control plane
    # ------------------------------------------------------------------

    def pin(
        self,
        ref: ContentRef,
        *,
        priority: int = 0,
        ttl_s: float | None = None,
        tier: str = "hbm",
    ) -> PinHandle:
        """Soft-pin content referenced by ``ref`` (returns a release handle).

        If the content has not been produced yet the entry stays pending and
        is activated by the first request declaring the same ``cache_key``.
        Over-budget content degrades to normal at activation time.
        """
        if not ref.cache_key:
            raise ValueError("pin requires a ContentRef with cache_key")
        if tier != "hbm":
            logger.debug("tier %r requested; external tier demotion is not implemented yet", tier)
        key = (ref.namespace, ref.cache_key)
        entry = self._entries.get(key)
        if entry is None:
            entry = LifecycleEntry(
                namespace=ref.namespace,
                cache_key=ref.cache_key,
                policy=_MODE_PIN,
                priority=priority,
                pin_count=0,
                ttl_s=ttl_s,
                expire_at=time.monotonic() + ttl_s if ttl_s else None,
                hashes=None,
                state=_STATE_PENDING,
            )
            self._entries[key] = entry
        entry.pin_count += 1
        handle = PinHandle.generate()
        self._handle_index[handle] = key
        self.metrics["pin_requests"] += 1
        return handle

    def set_ttl(self, ref: ContentRef | PinHandle | tuple[str, str], ttl_s: float | None) -> None:
        """Set/adjust/cancel (``None``) the TTL of a registered entry."""
        key = self._resolve_key(ref)
        entry = self._entries.get(key)
        if entry is None:
            logger.debug("set_ttl on unknown key %r, ignoring", key)
            return
        entry.ttl_s = ttl_s
        entry.expire_at = time.monotonic() + ttl_s if ttl_s else None
        self._recompute_next_expiry()

    def release(self, ref: ContentRef | PinHandle | tuple[str, str]) -> bool:
        """Release content early (handle-based decrements, key-based clears).

        Returns ``False`` when the reference is unknown (idempotent). The
        block-level release plan is available via ``take_release_plan``.
        """
        key = self._resolve_key(ref, required=False)
        if key is None:
            return False
        entry = self._entries.get(key)
        if entry is None:
            return False
        self.metrics["release_calls"] += 1
        if isinstance(ref, PinHandle):
            entry.pin_count = max(0, entry.pin_count - 1)
        else:
            entry.pin_count = 0
        if entry.pin_count > 0:
            return True
        self._release_plan.extend(self._release_plan_for(entry))
        self._unregister(key)
        return True

    def release_key(self, namespace: str, cache_key: str) -> bool:
        return self.release((namespace, cache_key))

    def take_release_plan(self) -> list[Any]:
        """Pop the pending block-level release plan (raw block hashes)."""
        plan = self._release_plan
        self._release_plan = []
        return plan

    def flush(self, keep_protected: bool, cached_hashes: Iterable[Any]) -> list[Any]:
        """Compute the flush plan over ``cached_hashes`` (raw block hashes)."""
        if not keep_protected:
            return list(cached_hashes)
        return [h for h in cached_hashes if self.protection_level(h) == 0]

    # ------------------------------------------------------------------
    # Scheduling-time queries (hot path)
    # ------------------------------------------------------------------

    def protection_level(self, block_hash: Any) -> int:
        """Max protection level of the hash; 0 means unprotected.

        The level is ``entry.priority + 1`` of the strongest active (non
        expired) entry covering the hash.
        """
        entry_keys = self._hash_index.get(block_hash)
        if not entry_keys:
            return 0
        now = time.monotonic()
        level = 0
        for entry_key in entry_keys:
            entry = self._entries.get(entry_key)
            if entry is None or entry.state != _STATE_ACTIVE or entry.hashes is None:
                continue
            if entry.expire_at is not None and entry.expire_at <= now:
                continue
            level = max(level, entry.priority + 1)
        return level

    def has_protection(self) -> bool:
        """Fast-path check: whether any hash is currently protected."""
        return bool(self._hash_index)

    def maybe_sweep(self) -> None:
        """Lazily downgrade expired entries; O(1) when nothing is due."""
        now = time.monotonic()
        if self._next_expiry is None or now < self._next_expiry:
            return
        expired = [
            key for key, entry in self._entries.items() if entry.expire_at is not None and entry.expire_at <= now
        ]
        for key in expired:
            self.metrics["ttl_expired"] += 1
            self._unregister(key)
        self._recompute_next_expiry()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_key(self, ref: Any, required: bool = True) -> tuple[str, str] | None:
        if isinstance(ref, PinHandle):
            key = self._handle_index.get(ref)
            if key is None and required:
                raise ValueError("unknown PinHandle")
            return key
        if isinstance(ref, ContentRef):
            if not ref.cache_key:
                if required:
                    raise ValueError("ContentRef requires cache_key")
                return None
            return (ref.namespace, ref.cache_key)
        if isinstance(ref, tuple) and len(ref) == 2:
            return (ref[0], ref[1])
        if isinstance(ref, str):
            return ("default", ref)
        if required:
            raise ValueError(f"cannot resolve content reference: {ref!r}")
        return None

    def _release_plan_for(self, entry: LifecycleEntry) -> list[Any]:
        if not entry.hashes:
            return []
        key = (entry.namespace, entry.cache_key)
        now = time.monotonic()
        plan = []
        for block_hash in entry.hashes:
            protected_by_others = False
            for other_key in self._hash_index.get(block_hash, ()):  # excludes self
                if other_key == key:
                    continue
                other = self._entries.get(other_key)
                if (
                    other is not None
                    and other.state == _STATE_ACTIVE
                    and (other.expire_at is None or other.expire_at > now)
                ):
                    protected_by_others = True
                    break
            if not protected_by_others:
                plan.append(block_hash)
        return plan

    def _quota_allows(self, num_new_blocks: int) -> bool:
        if self._kv_cache_manager is None:
            return True
        try:
            total = self._kv_cache_manager.block_pool.num_gpu_blocks
        except AttributeError:
            return True
        budget = envs.VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO * total
        return len(self._hash_index) + num_new_blocks <= budget

    def _index_hashes(self, key: tuple[str, str], hashes: frozenset) -> None:
        for block_hash in hashes:
            self._hash_index.setdefault(block_hash, set()).add(key)

    def _unindex_hashes(self, key: tuple[str, str], hashes: frozenset) -> None:
        for block_hash in hashes:
            entry_keys = self._hash_index.get(block_hash)
            if entry_keys is None:
                continue
            entry_keys.discard(key)
            if not entry_keys:
                del self._hash_index[block_hash]

    def _unregister(self, key: tuple[str, str]) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None and entry.hashes is not None:
            self._unindex_hashes(key, entry.hashes)
        self._recompute_next_expiry()

    def _recompute_next_expiry(self) -> None:
        expiries = [e.expire_at for e in self._entries.values() if e.expire_at is not None]
        self._next_expiry = min(expiries) if expiries else None
