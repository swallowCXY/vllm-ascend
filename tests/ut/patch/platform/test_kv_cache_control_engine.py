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

import asyncio
import importlib
import logging
import sys
import types
from importlib.util import spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_control_manager import KVCacheControlManager

_PATCH_MODULE = "vllm_ascend.patch.platform.patch_kv_cache_control_engine"
_PATCH_FILE = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_control_engine.py"
_ROUTER_MODULE = "vllm_ascend.entrypoints.kv_cache_router"
_ROUTER_FILE = Path(__file__).resolve().parents[4] / "vllm_ascend/entrypoints/kv_cache_router.py"
_FAKE_MODULES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.engine",
    "vllm.v1.engine.core",
    "vllm.v1.engine.async_llm",
    "vllm.logger",
    "vllm.entrypoints",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.api_server",
    "vllm_ascend.patch",
    "vllm_ascend.patch.platform",
    "fastapi",
    "pydantic",
)


class _FakeEngineCore:
    pass


class _FakeAsyncLLM:
    pass


def _fake_fastapi():
    class _FakeRouter:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def post(self, path):
            def deco(fn):
                self.routes.append((path, fn))
                return fn

            return deco

    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.APIRouter = _FakeRouter
    fastapi_mod.Request = object

    class _BaseModel:
        def __init__(self, **kwargs):
            for name in type(self).__annotations__:
                default = getattr(type(self), name, None)
                setattr(self, name, kwargs.get(name, default))

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = _BaseModel
    return fastapi_mod, pydantic_mod


def _build_fakes():
    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_mod = types.ModuleType("vllm.v1.core")
    kv_cache_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils_mod.get_block_hash = lambda block_hash: (
        block_hash[0] if isinstance(block_hash, tuple) else block_hash
    )
    kv_cache_utils_mod.make_block_hash_with_group_id = lambda block_hash, group_id: (block_hash, group_id)
    engine_mod = types.ModuleType("vllm.v1.engine")
    core_cls_mod = types.ModuleType("vllm.v1.engine.core")
    core_cls_mod.EngineCore = _FakeEngineCore
    async_llm_mod = types.ModuleType("vllm.v1.engine.async_llm")
    async_llm_mod.AsyncLLM = _FakeAsyncLLM
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.logger = logging.getLogger("vllm")
    entrypoints_mod = types.ModuleType("vllm.entrypoints")
    openai_mod = types.ModuleType("vllm.entrypoints.openai")
    api_server_mod = types.ModuleType("vllm.entrypoints.openai.api_server")
    api_server_mod.build_app = lambda *args, **kwargs: args[0] if args else kwargs.get("app")
    patch_pkg = types.ModuleType("vllm_ascend.patch")
    patch_pkg.__path__ = []
    platform_pkg = types.ModuleType("vllm_ascend.patch.platform")
    platform_pkg.__path__ = [str(_PATCH_FILE.parent)]
    fastapi_mod, pydantic_mod = _fake_fastapi()
    fake_map = dict(
        zip(
            _FAKE_MODULES,
            (
                vllm_mod,
                v1_mod,
                core_mod,
                kv_cache_utils_mod,
                engine_mod,
                core_cls_mod,
                async_llm_mod,
                logger_mod,
                entrypoints_mod,
                openai_mod,
                api_server_mod,
                patch_pkg,
                platform_pkg,
                fastapi_mod,
                pydantic_mod,
            ),
        )
    )
    for name, mod in fake_map.items():
        sys.modules[name] = mod
    return api_server_mod


@pytest.fixture()
def engine_env():
    saved = {name: sys.modules.get(name) for name in _FAKE_MODULES}
    api_server_mod = _build_fakes()
    loaded = {}
    try:
        spec = spec_from_file_location(_PATCH_MODULE, _PATCH_FILE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PATCH_MODULE] = mod
        spec.loader.exec_module(mod)
        loaded["patch"] = mod
        router_spec = spec_from_file_location(_ROUTER_MODULE, _ROUTER_FILE)
        router_mod = importlib.util.module_from_spec(router_spec)
        sys.modules[_ROUTER_MODULE] = router_mod
        router_spec.loader.exec_module(router_mod)
        loaded["router"] = router_mod
        loaded["api_server_mod"] = api_server_mod
        yield loaded
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop(_PATCH_MODULE, None)
        sys.modules.pop(_ROUTER_MODULE, None)


def _kvcm_with_entry(key="k", num=2):
    kvcm = KVCacheControlManager()
    req = SimpleNamespace(
        kv_transfer_params={"kv_cache_control": {"mode": "pin", "cache_key": key}},
        request_id=f"req-{key}",
        block_hashes=[f"h{i}".encode() for i in range(num)],
    )
    kvcm.on_request_finished(req)
    return kvcm


def _fake_block_pool():
    cached = {
        (b"h0", 0): SimpleNamespace(block_id=10),
        (b"h0", 1): SimpleNamespace(block_id=11),
        (b"h2", 0): SimpleNamespace(block_id=12),
    }
    evicted = []

    class _Map:
        def get_one_block(self, key):
            return cached.get(key)

        @property
        def _cache(self):
            return cached

    class _Pool:
        num_gpu_blocks = 1000
        cached_block_hash_to_block = _Map()

        @staticmethod
        def evict_blocks(block_ids):
            evicted.append(set(block_ids))

    return _Pool(), cached, evicted


def _fake_engine(kvcm):
    block_pool, cached, evicted = _fake_block_pool()
    manager = SimpleNamespace(
        block_pool=block_pool,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        reset_prefix_cache=lambda: True,
    )
    manager.kv_cache_control_manager = kvcm
    kvcm.bind_kv_cache_manager(manager)
    connector = SimpleNamespace(queue_external_delete=MagicMock())
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(kv_cache_manager=manager, connector=connector),
    )
    return engine, evicted


class TestEngineControlPlane:
    def test_engine_core_methods_attached(self, engine_env):
        for name in ("kv_cache_pin", "kv_cache_set_ttl", "kv_cache_release", "kv_cache_flush"):
            assert hasattr(_FakeEngineCore, name)

    def test_pin_forwards_to_manager(self, engine_env):
        kvcm = KVCacheControlManager()
        engine, _ = _fake_engine(kvcm)
        handle = _FakeEngineCore.kv_cache_pin(engine, "default", "k2", 3, 60.0, "hbm")
        assert isinstance(handle, str)
        assert kvcm.metrics["pin_requests"] == 1

    def test_release_evicts_and_queues_external_delete(self, engine_env):
        kvcm = _kvcm_with_entry("k")
        engine, evicted = _fake_engine(kvcm)
        released = _FakeEngineCore.kv_cache_release(engine, "default", "k")
        assert released is True
        assert evicted == [{10, 11}]
        kvcm_conn = engine.scheduler.connector.queue_external_delete
        kvcm_conn.assert_called_once()
        assert set(kvcm_conn.call_args[0][0]) == {b"h0", b"h1"}

    def test_flush_keep_protected(self, engine_env):
        kvcm = _kvcm_with_entry("k")
        engine, evicted = _fake_engine(kvcm)
        evicted_blocks = _FakeEngineCore.kv_cache_flush(engine, True)
        assert evicted_blocks == 1
        assert evicted == [{12}]

    def test_flush_without_protection_resets(self, engine_env):
        engine, evicted = _fake_engine(KVCacheControlManager())
        assert _FakeEngineCore.kv_cache_flush(engine, True) == 3
        assert _FakeEngineCore.kv_cache_flush(engine, False) == -1

    def test_async_llm_entry_point(self, engine_env):
        calls = []

        class _Client:
            async def call_utility_async(self, method, *args):
                calls.append((method, args))
                return "OK"

        engine_llm = _FakeAsyncLLM()
        engine_llm.engine_core = _Client()
        result = asyncio.run(
            _FakeAsyncLLM.kv_cache_control_async(engine_llm, "kv_cache_pin", "ns", "k", 0, None, "hbm")
        )
        assert result == "OK"
        assert calls[0][0] == "kv_cache_pin"

    def test_build_app_mounts_router(self, engine_env):
        api_server_mod = engine_env["api_server_mod"]
        app = SimpleNamespace(include_router=MagicMock())
        result = api_server_mod.build_app(app)
        assert result is app
        app.include_router.assert_called_once()


class TestRouterEndpoints:
    def test_pin_endpoint(self, engine_env):
        kvcm = KVCacheControlManager()
        engine, _ = _fake_engine(kvcm)
        client = SimpleNamespace(
            call_utility=MagicMock(return_value="HANDLE"),
        )
        raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=client)))
        req = engine_env["router"].PinRequest(cache_key="k", priority=2)
        resp = asyncio.run(engine_env["router"].pin(req, raw))
        assert resp == {"handle": "HANDLE"}
        client.call_utility.assert_called_once_with("kv_cache_pin", "default", "k", 2, None, "hbm")

    def test_release_endpoint(self, engine_env):
        kvcm = _kvcm_with_entry("k")
        engine, _ = _fake_engine(kvcm)
        calls = []

        class _Client:
            async def call_utility_async(self, method, *args):
                calls.append(method)
                return _FakeEngineCore.kv_cache_release(engine, *args)

        raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=_Client())))
        req = engine_env["router"].KeyRequest(cache_key="k")
        resp = asyncio.run(engine_env["router"].release(req, raw))
        assert resp == {"released": True}
        assert calls == ["kv_cache_release"]

    def test_flush_endpoint(self, engine_env):
        client = SimpleNamespace(
            call_utility=MagicMock(return_value=5),
        )
        raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=client)))
        req = engine_env["router"].FlushRequest(keep_protected=False)
        resp = asyncio.run(engine_env["router"].flush(req, raw))
        assert resp == {"evicted_blocks": 5}
        client.call_utility.assert_called_once_with("kv_cache_flush", False)
