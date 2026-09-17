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
    "vllm.v1.core.kv_cache_manager",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.engine",
    "vllm.v1.engine.core",
    "vllm.v1.engine.async_llm",
    "vllm.logger",
    "vllm.entrypoints",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.api_server",
    "vllm.entrypoints.openai.chat_completion",
    "vllm.entrypoints.openai.chat_completion.serving",
    "vllm_ascend.patch",
    "vllm_ascend.patch.platform",
    "fastapi",
    "pydantic",
)


class _FakeEngineCore:
    pass


class _FakeAsyncLLM:
    pass


class _FakeServingChat:
    async def render_chat_request(self, request):
        return "RENDERED"

    async def create_chat_completion(self, request, raw_request=None):
        return SimpleNamespace(mark="RESPONSE")


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

    class _FakeKVCacheManagerForImport:
        def allocate_slots(self, *args, **kwargs):
            return None

        def cache_blocks(self, *args, **kwargs):
            return None

        def free(self, *args, **kwargs):
            return None

    mgr_mod = types.ModuleType("vllm.v1.core.kv_cache_manager")
    mgr_mod.KVCacheManager = _FakeKVCacheManagerForImport
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
    chat_pkg = types.ModuleType("vllm.entrypoints.openai.chat_completion")
    serving_mod = types.ModuleType("vllm.entrypoints.openai.chat_completion.serving")
    serving_mod.OpenAIServingChat = _FakeServingChat
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
                mgr_mod,
                kv_cache_utils_mod,
                engine_mod,
                core_cls_mod,
                async_llm_mod,
                logger_mod,
                entrypoints_mod,
                openai_mod,
                api_server_mod,
                chat_pkg,
                serving_mod,
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
    try:
        spec = spec_from_file_location(_PATCH_MODULE, _PATCH_FILE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PATCH_MODULE] = mod
        spec.loader.exec_module(mod)
        loaded = {"patch": mod, "api_server_mod": api_server_mod}
        router_spec = spec_from_file_location(_ROUTER_MODULE, _ROUTER_FILE)
        router_mod = importlib.util.module_from_spec(router_spec)
        sys.modules[_ROUTER_MODULE] = router_mod
        router_spec.loader.exec_module(router_mod)
        loaded["router"] = router_mod
        yield loaded
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop(_PATCH_MODULE, None)
        sys.modules.pop(_ROUTER_MODULE, None)


def _kvcm_with_history(request_id="r1", num=2):
    kvcm = KVCacheControlManager()
    req = SimpleNamespace(
        kv_transfer_params=None,
        request_id=request_id,
        block_hashes=[f"h{i}".encode() for i in range(num)],
    )
    kvcm.on_request_finished(req)
    return kvcm


def _fake_engine(kvcm):
    evicted = []
    cached = {
        (b"h0", 0): SimpleNamespace(block_id=10),
        (b"h0", 1): SimpleNamespace(block_id=11),
    }

    class _Map:
        def get_one_block(self, key):
            return cached.get(key)

    class _Pool:
        num_gpu_blocks = 1000
        cached_block_hash_to_block = _Map()

        @staticmethod
        def evict_blocks(block_ids):
            evicted.append(set(block_ids))

    manager = SimpleNamespace(
        block_pool=_Pool(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        kv_cache_control_manager=kvcm,
    )
    kvcm.bind_kv_cache_manager(manager)
    engine = SimpleNamespace(scheduler=SimpleNamespace(kv_cache_manager=manager))
    return engine, evicted


class TestEngineReleaseControlPlane:
    def test_release_method_attached(self, engine_env):
        assert hasattr(_FakeEngineCore, "kv_cache_release")
        assert not hasattr(_FakeEngineCore, "kv_cache_pin")
        assert not hasattr(_FakeEngineCore, "kv_cache_flush")

    def test_release_by_request_id(self, engine_env):
        kvcm = _kvcm_with_history("r1")
        engine, evicted = _fake_engine(kvcm)
        released = _FakeEngineCore.kv_cache_release(engine, "r1")
        assert released is True
        assert evicted == [{10, 11}]
        assert kvcm.take_release_plan() == []

    def test_release_unknown_request(self, engine_env):
        engine, evicted = _fake_engine(KVCacheControlManager())
        assert _FakeEngineCore.kv_cache_release(engine, "ghost") is False
        assert evicted == []

    def test_async_llm_entry_point(self, engine_env):
        calls = []

        class _Client:
            async def call_utility_async(self, method, *args):
                calls.append((method, args))
                return "OK"

        engine_llm = _FakeAsyncLLM()
        engine_llm.engine_core = _Client()
        result = asyncio.run(_FakeAsyncLLM.kv_cache_control_async(engine_llm, "kv_cache_release", "r1"))
        assert result == "OK"
        assert calls == [("kv_cache_release", ("r1",))]

    def test_build_app_mounts_router(self, engine_env):
        api_server_mod = engine_env["api_server_mod"]
        app = SimpleNamespace(include_router=MagicMock())
        result = api_server_mod.build_app(app)
        assert result is app
        app.include_router.assert_called_once()

    def test_chat_serving_wrapped(self, engine_env):
        assert getattr(_FakeServingChat.render_chat_request, "__vcc_router_attached__", False)
        assert getattr(_FakeServingChat.create_chat_completion, "__vcc_router_attached__", False)


class TestRouterReleaseOnly:
    def test_release_endpoint(self, engine_env):
        kvcm = _kvcm_with_history("r1")
        engine, _ = _fake_engine(kvcm)
        calls = []

        class _Client:
            async def call_utility_async(self, method, *args):
                calls.append(method)
                return _FakeEngineCore.kv_cache_release(engine, *args)

        raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=_Client())))
        req = engine_env["router"].ReleaseRequest(request_id="r1")
        resp = asyncio.run(engine_env["router"].release(req, raw))
        assert resp == {"released": True}
        assert calls == ["kv_cache_release"]

    def test_router_has_only_release(self, engine_env):
        paths = [path for path, _ in engine_env["router"].router.routes]
        assert paths == ["/release"]
