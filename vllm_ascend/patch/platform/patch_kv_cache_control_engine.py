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
"""Engine-side control plane for KV cache lifecycle management.

The only out-of-band operation is release by ``request_id`` (out-of-band
modification of pin/TTL state is not supported). Adds a ``kv_cache_release``
utility method to ``EngineCore`` (resolved by the existing ``call_utility``
reflection channel), an ``AsyncLLM`` entry point, and mounts the
``/kv_cache/release`` HTTP router onto the OpenAI API server by wrapping
``build_app``.
"""

from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore

from vllm_ascend import envs
from vllm_ascend.patch.platform.patch_kv_cache_control import evict_hashes

_BUILD_APP_MARKER = "__vcc_router_attached__"


def _get_context(engine_core: EngineCore) -> tuple[Any, Any]:
    scheduler = engine_core.scheduler
    manager = getattr(scheduler, "kv_cache_manager", None)
    kvcm = getattr(manager, "kv_cache_control_manager", None)
    if kvcm is None:
        raise RuntimeError("KV cache control is disabled (VLLM_ASCEND_KV_CACHE_CONTROL)")
    return manager, kvcm


def _engine_kv_cache_release(self: EngineCore, request_id: str) -> bool:
    manager, kvcm = _get_context(self)
    released = kvcm.release_request(request_id)
    if released:
        plan = kvcm.take_release_plan()
        evicted = evict_hashes(manager, plan)
        logger.info(
            "KV cache release request_id=%r unregistered %d blocks",
            request_id,
            evicted,
        )
    return released


def _patch_engine_core() -> None:
    EngineCore.kv_cache_release = _engine_kv_cache_release  # type: ignore[attr-defined]


async def _kv_cache_control_async(self: AsyncLLM, method: str, *args: Any) -> Any:
    return await self.engine_core.call_utility_async(method, *args)


def _patch_async_llm() -> None:
    AsyncLLM.kv_cache_control_async = _kv_cache_control_async  # type: ignore[attr-defined]


def _patch_api_server() -> None:
    try:
        from vllm.entrypoints.openai import api_server as api_server_mod
    except ImportError:
        logger.warning("OpenAI api_server not importable; /kv_cache routes not mounted")
        return
    original = api_server_mod.build_app
    if getattr(original, _BUILD_APP_MARKER, False):
        return

    @wraps(original)
    def _patched_build_app(*args: Any, **kwargs: Any) -> Any:
        app = original(*args, **kwargs)
        try:
            from vllm_ascend.entrypoints.kv_cache_router import attach_router

            attach_router(app)
        except Exception:
            logger.exception("Failed to attach /kv_cache router")
        return app

    _patched_build_app.__vcc_router_attached__ = True  # type: ignore[attr-defined]
    api_server_mod.build_app = _patched_build_app


def _patch_chat_serving() -> None:
    """Message-level declaration extraction + response status attachment."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    from vllm_ascend.entrypoints.kv_cache_message import (
        attach_status_to_response,
        inject_request_control,
    )

    if getattr(OpenAIServingChat.render_chat_request, _BUILD_APP_MARKER, False):
        return

    _original_render = OpenAIServingChat.render_chat_request

    @wraps(_original_render)
    async def _patched_render_chat_request(self: Any, request: Any) -> Any:
        inject_request_control(self, request)
        return await _original_render(self, request)

    _patched_render_chat_request.__vcc_router_attached__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.render_chat_request = _patched_render_chat_request

    _original_create = OpenAIServingChat.create_chat_completion

    @wraps(_original_create)
    async def _patched_create_chat_completion(self: Any, request: Any, raw_request: Any = None) -> Any:
        result = await _original_create(self, request, raw_request)
        if raw_request is not None and not isinstance(result, AsyncGenerator):
            attach_status_to_response(request, result)
        return result

    _patched_create_chat_completion.__vcc_router_attached__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.create_chat_completion = _patched_create_chat_completion
    logger.info("KV cache control enabled: message-level declaration handling applied")


def _apply_patch() -> None:
    if not envs.VLLM_ASCEND_KV_CACHE_CONTROL:
        return
    _patch_engine_core()
    _patch_async_llm()
    _patch_api_server()
    _patch_chat_serving()
    logger.info("KV cache control enabled: release control plane and /kv_cache/release route applied")


_apply_patch()
