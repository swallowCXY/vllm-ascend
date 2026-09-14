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
metadata only: no KV tensors, no block objects, and it never mutates the
block pool directly. See docs:
docs/source/developer_guide/Design_Documents/KV_Cache_Control_Manager.md
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = getLogger(__name__)

_KV_CACHE_CONTROL_KEY = "kv_cache_control"
_MODE_NO_STORE = "no_store"
_MODE_PIN = "pin"
_MODE_TTL = "ttl"
_PARSED_ATTR = "_kv_cache_control_parsed"
_UNPARSED = object()


@dataclass(frozen=True)
class ContentRef:
    """Agent-facing reference to cacheable content.

    One of ``cache_key`` / ``request_id`` must be provided. ``prefix_tokens``
    optionally limits the reference to a token-count prefix (floored to the
    engine block size by the manager).
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


class KVCacheControlManager:
    """Holds lifecycle declarations and answers scheduling-time queries.

    The manager is attached to ``KVCacheManager`` (scheduler side) and to the
    AscendStore connector scheduler. Hot-path queries are O(1) and memoized
    per request.
    """

    def __init__(self) -> None:
        self.metrics: dict[str, int] = {
            "no_store_requests": 0,
            "unsupported_requests": 0,
            "parse_errors": 0,
        }

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
            self.metrics["unsupported_requests"] += 1
            logger.warning(
                "kv_cache_control mode %r is declared but not supported yet, ignoring",
                mode,
            )
            return None
        self.metrics["unsupported_requests"] += 1
        logger.debug("Unknown kv_cache_control mode %r, ignoring", mode)
        return None

    def pin(
        self,
        ref: ContentRef,
        *,
        priority: int = 0,
        ttl_s: float | None = None,
        tier: str = "hbm",
    ) -> PinHandle:
        """Soft-pin content referenced by ``ref`` (design §14.1, phase P1)."""
        raise NotImplementedError("pin will be implemented in phase P1")

    def set_ttl(self, ref: ContentRef | PinHandle, ttl_s: float | None) -> None:
        """Set/adjust the TTL of content (design §14.2, phase P1)."""
        raise NotImplementedError("set_ttl will be implemented in phase P1")

    def release(self, ref: ContentRef | PinHandle) -> bool:
        """Release content early (design §14.3, phase P2)."""
        raise NotImplementedError("release will be implemented in phase P2")
