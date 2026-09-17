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
"""KV cache lifecycle control patches (no-store / pin / release).

Engine side:

- Requests declaring ``mode: no_store`` never register their blocks into the
  prefix cache: ``allocate_slots`` is forced into the upstream
  ``delay_cache_blocks`` path (the same mechanism used by P/D async load) and
  explicit ``KVCacheManager.cache_blocks`` calls become no-ops.
- Requests declaring ``mode: pin`` get their prefix hard-protected at finish
  (see ``patch_kv_cache_eviction``); ``mode: release`` unregisters all of the
  request's prefix-cache registrations via ``evict_hashes``.
- ``allocate_slots`` converts ``PinProtectedExhaustedError`` into an
  allocation failure (``None``) so the scheduler preempts normally instead of
  evicting pinned content.

Serving side:

- ``OpenAIServingChat.render_chat_request`` extracts message-level
  ``kv_cache_control`` declarations, adjudicates conflicts and computes pin
  boundaries (see ``entrypoints/kv_cache_message``) before the engine inputs
  are built; ``create_chat_completion`` attaches the adjudication outcome to
  non-streaming responses as ``kv_cache_control_status``.

Requests without declarations follow the original code path with one dict
lookup of overhead.
"""

from collections.abc import AsyncGenerator
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
    _patch_chat_serving()
    logger.info("KV cache control enabled: no-store / pin / release support applied")


def _patch_chat_serving() -> None:
    """Message-level declaration extraction + response status attachment."""
    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except ImportError:
        logger.warning("OpenAI chat serving not importable; message-level declarations inactive")
        return
    from vllm_ascend.entrypoints.kv_cache_message import (
        attach_status_to_response,
        inject_request_control,
    )

    if getattr(OpenAIServingChat.render_chat_request, _PATCH_MARKER, False):
        return

    _original_render = OpenAIServingChat.render_chat_request

    @wraps(_original_render)
    async def _patched_render_chat_request(self: Any, request: Any) -> Any:
        inject_request_control(self, request)
        return await _original_render(self, request)

    _patched_render_chat_request.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.render_chat_request = _patched_render_chat_request

    _original_create = OpenAIServingChat.create_chat_completion

    @wraps(_original_create)
    async def _patched_create_chat_completion(self: Any, request: Any, raw_request: Any = None) -> Any:
        result = await _original_create(self, request, raw_request)
        if raw_request is not None and not isinstance(result, AsyncGenerator):
            attach_status_to_response(request, result)
        return result

    _patched_create_chat_completion.__vcc_no_store_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.create_chat_completion = _patched_create_chat_completion
    logger.info("KV cache control enabled: message-level declaration handling applied")


_apply_patch()
