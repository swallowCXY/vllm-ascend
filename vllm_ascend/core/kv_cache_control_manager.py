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

KVCacheControlManager (KVCM) holds message-level lifecycle declarations and
answers scheduling-time queries. Three modes, mutually exclusive per request:

- ``pin``: hard protection of the request prefix (tools + messages up to the
  declared message) until TTL expiry; never evicted while protected.
- ``release``: on request finish, unregister ALL prefix-cache registrations
  covered by the request's block hashes.
- ``no_store``: never register the request's newly produced blocks.

It owns metadata only: no KV tensors, no block objects. Effects on the block
pool are applied by callers (patches / engine control methods). See:
docs/source/developer_guide/Design_Documents/KV_Cache_Control_Manager.md
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

from vllm_ascend import envs

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = getLogger(__name__)

_KV_CACHE_CONTROL_KEY = "kv_cache_control"
_MODE_NO_STORE = "no_store"
_MODE_PIN = "pin"
_MODE_RELEASE = "release"
_SUPPORTED_MODES = (_MODE_NO_STORE, _MODE_PIN, _MODE_RELEASE)
_PARSED_ATTR = "_kv_cache_control_parsed"
_UNPARSED = object()


class PinProtectedExhaustedError(Exception):
    """Raised when free blocks are all hard-protected and allocation cannot
    proceed without evicting pinned content. The allocate_slots wrapper
    converts this into an allocation failure (None) so the scheduler takes
    its normal preemption path."""


@dataclass
class PinEntry:
    """Hard-protection entry for one finished request's prefix blocks."""

    request_id: str
    hashes: frozenset
    expire_at: float


class KVCacheControlManager:
    """Holds lifecycle declarations and answers scheduling-time queries.

    One instance is attached to ``KVCacheManager`` (scheduler side, bound to
    the block pool for quota accounting) and one to the AscendStore connector
    scheduler (no-store store skip). Hot-path queries are O(1) and memoized
    per request.
    """

    def __init__(self) -> None:
        self.metrics: dict[str, int] = {
            "no_store_requests": 0,
            "pin_requests": 0,
            "release_requests": 0,
            "quota_degraded": 0,
            "ttl_expired": 0,
            "parse_errors": 0,
        }
        self._pin_entries: dict[str, PinEntry] = {}
        self._hash_index: dict[Any, set[str]] = {}
        self._next_expiry: float | None = None
        self._release_plan: list[Any] = []
        self._kv_cache_manager: Any = None
        self._block_size: int | None = None

    def bind_kv_cache_manager(self, kv_cache_manager: Any, block_size: int | None = None) -> None:
        """Bind the owning KVCacheManager for quota accounting."""
        self._kv_cache_manager = kv_cache_manager
        self._block_size = block_size

    # ------------------------------------------------------------------
    # Request-level declaration parsing (aggregated by the serving layer)
    # ------------------------------------------------------------------

    def parse_request_control(self, request: Request) -> dict[str, Any] | None:
        """Parse and memoize the request-level ``kv_cache_control`` declaration."""
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

    def is_release(self, request: Request) -> bool:
        """Whether the request declared ``mode: release``."""
        control = self.parse_request_control(request)
        return control is not None and control["mode"] == _MODE_RELEASE

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
        if not isinstance(raw, dict) or raw.get("mode") not in _SUPPORTED_MODES:
            self.metrics["parse_errors"] += 1
            logger.warning("Invalid kv_cache_control payload, ignoring: %r", raw)
            return None
        mode = raw["mode"]
        if mode == _MODE_NO_STORE:
            self.metrics["no_store_requests"] += 1
            return {"mode": _MODE_NO_STORE}
        if mode == _MODE_RELEASE:
            self.metrics["release_requests"] += 1
            return {"mode": _MODE_RELEASE}
        boundary = raw.get("pin_boundary_tokens")
        ttl_s = raw.get("ttl_s")
        if ttl_s is None:
            ttl_s = envs.VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S
        self.metrics["pin_requests"] += 1
        return {
            "mode": _MODE_PIN,
            "ttl_s": float(ttl_s),
            "pin_boundary_tokens": boundary if isinstance(boundary, int) else None,
        }

    # ------------------------------------------------------------------
    # Request finish hooks (called from the KVCacheManager.free wrapper)
    # ------------------------------------------------------------------

    def on_request_finished(self, request: Request) -> None:
        """Activate pin / queue release plan at request finish.

        Called before the request's blocks are returned to the free queue.
        """
        control = self.parse_request_control(request)
        if control is None:
            return
        hashes = list(getattr(request, "block_hashes", None) or [])
        if control["mode"] == _MODE_PIN:
            self._activate_pin(request, hashes, control)
        elif control["mode"] == _MODE_RELEASE and hashes:
            self._release_plan.extend(hashes)

    def _activate_pin(self, request: Request, hashes: list[Any], control: dict[str, Any]) -> None:
        request_id = request.request_id
        boundary_tokens = control.get("pin_boundary_tokens")
        if boundary_tokens is not None and self._block_size:
            hashes = hashes[: max(0, boundary_tokens // self._block_size)]
        if not hashes:
            logger.debug("Pin declaration for request %s covers no full blocks, ignoring", request_id)
            return
        if not self._quota_allows(len(hashes)):
            self.metrics["quota_degraded"] += 1
            logger.warning(
                "Pin for request %s exceeds pin budget (%d blocks), deactivating: content degrades to normal caching",
                request_id,
                len(hashes),
            )
            return
        self._unregister(request_id)
        entry = PinEntry(
            request_id=request_id,
            hashes=frozenset(hashes),
            expire_at=time.monotonic() + control["ttl_s"],
        )
        self._pin_entries[request_id] = entry
        for block_hash in entry.hashes:
            self._hash_index.setdefault(block_hash, set()).add(request_id)
        self._recompute_next_expiry()

    # ------------------------------------------------------------------
    # Release plan (consumed by the KVCacheManager.free wrapper)
    # ------------------------------------------------------------------

    def take_release_plan(self) -> list[Any]:
        """Pop the pending block-level release plan (raw block hashes)."""
        plan = self._release_plan
        self._release_plan = []
        return plan

    # ------------------------------------------------------------------
    # Scheduling-time queries (hot path)
    # ------------------------------------------------------------------

    def protection_level(self, block_hash: Any) -> int:
        """1 if the hash is hard-protected by an unexpired pin, else 0."""
        request_ids = self._hash_index.get(block_hash)
        if not request_ids:
            return 0
        now = time.monotonic()
        for request_id in request_ids:
            entry = self._pin_entries.get(request_id)
            if entry is not None and entry.expire_at > now:
                return 1
        return 0

    def has_protection(self) -> bool:
        """Fast-path check: whether any hash is currently protected."""
        return bool(self._hash_index)

    def maybe_sweep(self) -> None:
        """Lazily drop expired pin entries; O(1) when nothing is due."""
        now = time.monotonic()
        if self._next_expiry is None or now < self._next_expiry:
            return
        expired = [request_id for request_id, entry in self._pin_entries.items() if entry.expire_at <= now]
        for request_id in expired:
            self.metrics["ttl_expired"] += 1
            self._unregister(request_id)
        self._recompute_next_expiry()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _quota_allows(self, num_new_blocks: int) -> bool:
        if self._kv_cache_manager is None:
            return True
        try:
            total = self._kv_cache_manager.block_pool.num_gpu_blocks
        except AttributeError:
            return True
        budget = envs.VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO * total
        return len(self._hash_index) + num_new_blocks <= budget

    def _unregister(self, request_id: str) -> None:
        entry = self._pin_entries.pop(request_id, None)
        if entry is None:
            return
        for block_hash in entry.hashes:
            request_ids = self._hash_index.get(block_hash)
            if request_ids is None:
                continue
            request_ids.discard(request_id)
            if not request_ids:
                del self._hash_index[block_hash]
        self._recompute_next_expiry()

    def _recompute_next_expiry(self) -> None:
        expiries = [entry.expire_at for entry in self._pin_entries.values()]
        self._next_expiry = min(expiries) if expiries else None
