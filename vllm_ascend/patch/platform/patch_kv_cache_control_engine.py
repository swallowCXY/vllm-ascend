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

Adds utility methods to ``EngineCore`` (resolved by the existing
``call_utility`` reflection channel), an ``AsyncLLM`` entry point, and mounts
the ``/kv_cache/*`` HTTP router onto the OpenAI API server by wrapping
``build_app``. Methods:

- ``kv_cache_pin(namespace, cache_key, priority, ttl_s, tier) -> str``
- ``kv_cache_set_ttl(namespace, cache_key, ttl_s) -> None``
- ``kv_cache_release(namespace, cache_key) -> bool``
- ``kv_cache_flush(keep_protected) -> int``
"""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import get_block_hash, make_block_hash_with_group_id
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_control_manager import ContentRef

_BUILD_APP_MARKER = "__vcc_router_attached__"


def _get_context(engine_core: EngineCore) -> tuple[Any, Any]:
    scheduler = engine_core.scheduler
    manager = getattr(scheduler, "kv_cache_manager", None)
    kvcm = getattr(manager, "kv_cache_control_manager", None)
    if kvcm is None:
        raise RuntimeError("KV cache control is disabled (VLLM_ASCEND_KV_CACHE_CONTROL)")
    return manager, kvcm


def _evict_hashes(manager: Any, hashes: list[Any]) -> int:
    """Resolve raw block hashes to block ids across groups and evict them."""
    block_pool = manager.block_pool
    num_groups = len(manager.kv_cache_config.kv_cache_groups)
    block_ids: set[int] = set()
    for block_hash in hashes:
        for group_id in range(num_groups):
            block = block_pool.cached_block_hash_to_block.get_one_block(
                make_block_hash_with_group_id(block_hash, group_id)
            )
            if block is not None:
                block_ids.add(block.block_id)
    if block_ids:
        block_pool.evict_blocks(block_ids)
    return len(block_ids)


def _queue_external_delete(engine_core: EngineCore, hashes: list[Any]) -> None:
    connector = getattr(engine_core.scheduler, "connector", None)
    queue = getattr(connector, "queue_external_delete", None)
    if queue is None:
        return
    try:
        queue(hashes)
    except Exception:
        logger.exception("Failed to queue external delete for %d hashes", len(hashes))


def _engine_kv_cache_pin(
    self: EngineCore,
    namespace: str,
    cache_key: str,
    priority: int,
    ttl_s: float | None,
    tier: str,
) -> str:
    _, kvcm = _get_context(self)
    ref = ContentRef(namespace=namespace, cache_key=cache_key)
    return str(kvcm.pin(ref, priority=priority, ttl_s=ttl_s, tier=tier))


def _engine_kv_cache_set_ttl(self: EngineCore, namespace: str, cache_key: str, ttl_s: float | None) -> None:
    _, kvcm = _get_context(self)
    kvcm.set_ttl((namespace, cache_key), ttl_s)


def _engine_kv_cache_release(self: EngineCore, namespace: str, cache_key: str) -> bool:
    manager, kvcm = _get_context(self)
    released = kvcm.release_key(namespace, cache_key)
    plan = kvcm.take_release_plan()
    if plan:
        evicted = _evict_hashes(manager, plan)
        logger.info(
            "KV cache release key=%r:%r evicted %d blocks from prefix cache",
            namespace,
            cache_key,
            evicted,
        )
        _queue_external_delete(self, plan)
    return released


def _engine_kv_cache_flush(self: EngineCore, keep_protected: bool) -> int:
    manager, kvcm = _get_context(self)
    if not keep_protected:
        return -1 if manager.reset_prefix_cache() else 0
    cached_map = manager.block_pool.cached_block_hash_to_block
    cached_hashes = [get_block_hash(key) for key in list(getattr(cached_map, "_cache", {}))]
    plan = kvcm.flush(keep_protected=True, cached_hashes=cached_hashes)
    if not plan:
        return 0
    return _evict_hashes(manager, plan)


def _patch_engine_core() -> None:
    EngineCore.kv_cache_pin = _engine_kv_cache_pin  # type: ignore[attr-defined]
    EngineCore.kv_cache_set_ttl = _engine_kv_cache_set_ttl  # type: ignore[attr-defined]
    EngineCore.kv_cache_release = _engine_kv_cache_release  # type: ignore[attr-defined]
    EngineCore.kv_cache_flush = _engine_kv_cache_flush  # type: ignore[attr-defined]


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


def _apply_patch() -> None:
    if not envs.VLLM_ASCEND_KV_CACHE_CONTROL:
        return
    _patch_engine_core()
    _patch_async_llm()
    _patch_api_server()
    logger.info("KV cache control enabled: engine control plane and /kv_cache routes applied")


_apply_patch()
