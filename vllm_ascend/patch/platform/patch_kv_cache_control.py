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
"""Request-level KV cache ``no_store`` support.

Requests declaring ``kv_transfer_params={"kv_cache_control": {"mode":
"no_store"}}`` never register their blocks into the prefix cache:

- ``allocate_slots`` is forced into the upstream ``delay_cache_blocks`` path
  (the same mechanism used by P/D async load), and
- explicit ``KVCacheManager.cache_blocks`` calls become no-ops.

Requests without the declaration (the vast majority) follow the original code
path with one dict lookup of overhead. Shared prefix blocks registered by
other requests are never touched, matching Claude Code's ``skipCacheWrite``
semantics: read cache hits still apply, only new writes are skipped.
"""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_control_manager import (
    KVCacheControlManager,
    PinProtectedExhaustedError,
)
from vllm_ascend.core.kv_cache_evict_utils import evict_hashes

# Positional index of ``delay_cache_blocks`` in ``allocate_slots`` after
# ``self`` and ``request`` (see vllm/v1/core/kv_cache_manager.py).
_DELAY_CACHE_BLOCKS_POS = 5
_PATCH_MARKER = "__vcc_no_store_patched__"


def _force_delay_cache_blocks(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > _DELAY_CACHE_BLOCKS_POS:
        return args[:_DELAY_CACHE_BLOCKS_POS] + (True,) + args[_DELAY_CACHE_BLOCKS_POS + 1 :], kwargs
    kwargs["delay_cache_blocks"] = True
    return args, kwargs


def _apply_patch() -> None:
    if not envs.VLLM_ASCEND_KV_CACHE_CONTROL:
        logger.info("VLLM_ASCEND_KV_CACHE_CONTROL is disabled, skip no-store patch")
        return
    if getattr(KVCacheManager.allocate_slots, _PATCH_MARKER, False):
        return

    _original_init = KVCacheManager.__init__

    @wraps(_original_init)
    def _patched_init(self: KVCacheManager, *args: Any, **kwargs: Any) -> None:
        _original_init(self, *args, **kwargs)
        kvcm = KVCacheControlManager()
        self.kv_cache_control_manager = kvcm
        block_pool = getattr(self, "block_pool", None)
        if block_pool is not None:
            block_pool.kv_cache_control_manager = kvcm
        block_size = None
        try:
            kv_cache_config = getattr(self, "kv_cache_config", None)
            if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
                block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        except AttributeError:
            block_size = None
        kvcm.bind_kv_cache_manager(self, block_size)

    _original_allocate_slots = KVCacheManager.allocate_slots

    @wraps(_original_allocate_slots)
    def _patched_allocate_slots(self: KVCacheManager, request: Any, *args: Any, **kwargs: Any) -> Any:
        kvcm = getattr(self, "kv_cache_control_manager", None)
        if kvcm is not None:
            kvcm.maybe_sweep()
            if kvcm.is_no_store(request):
                args, kwargs = _force_delay_cache_blocks(args, kwargs)
        try:
            return _original_allocate_slots(self, request, *args, **kwargs)
        except PinProtectedExhaustedError:
            # All free blocks are hard-protected; fail this allocation so the
            # scheduler takes its normal preemption path instead of evicting
            # pinned content.
            return None

    _original_cache_blocks = KVCacheManager.cache_blocks

    @wraps(_original_cache_blocks)
    def _patched_cache_blocks(self: KVCacheManager, request: Any, num_computed_tokens: int) -> None:
        kvcm = getattr(self, "kv_cache_control_manager", None)
        if kvcm is not None and kvcm.is_no_store(request):
            return
        return _original_cache_blocks(self, request, num_computed_tokens)

    _original_free = KVCacheManager.free

    @wraps(_original_free)
    def _patched_free(self: KVCacheManager, request: Any) -> None:
        kvcm = getattr(self, "kv_cache_control_manager", None)
        if kvcm is not None:
            kvcm.on_request_finished(request)
            plan = kvcm.take_release_plan()
            if plan:
                evicted = evict_hashes(self, plan)
                logger.info(
                    "KV cache release for request %s unregistered %d blocks",
                    getattr(request, "request_id", "?"),
                    evicted,
                )
        return _original_free(self, request)

    _patched_init.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]
    _patched_allocate_slots.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]
    _patched_cache_blocks.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]
    _patched_free.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]

    KVCacheManager.__init__ = _patched_init
    KVCacheManager.allocate_slots = _patched_allocate_slots
    KVCacheManager.cache_blocks = _patched_cache_blocks
    KVCacheManager.free = _patched_free
    logger.info("KV cache control enabled: no-store / pin / ttl support applied")


_apply_patch()
