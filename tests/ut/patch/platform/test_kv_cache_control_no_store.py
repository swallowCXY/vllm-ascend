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

_PATCH_MODULE = "vllm_ascend.patch.platform.patch_kv_cache_control"
_PATCH_FILE = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_control.py"
_MGR_MODULE = "vllm.v1.core.kv_cache_manager"
_MARKER = "__vcc_no_store_patched__"
_FAKE_MODULES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.core",
    _MGR_MODULE,
    "vllm.logger",
    "vllm.logging_utils",
    "vllm_ascend.patch",
    "vllm_ascend.patch.platform",
)

_calls = {"allocate": [], "cache_blocks": [], "free": []}


class _FakeKVCacheManager:
    def __init__(self, *args, **kwargs):
        self.kv_cache_control_manager = None

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        _calls["allocate"].append((args, kwargs))
        return "ALLOCATED"

    def cache_blocks(self, request, num_computed_tokens):
        _calls["cache_blocks"].append(num_computed_tokens)

    def free(self, request):
        _calls["free"].append(request.request_id)


def _no_store_request():
    return SimpleNamespace(
        kv_transfer_params={"kv_cache_control": {"mode": "no_store"}},
        request_id="r1",
    )


def _pin_request():
    return SimpleNamespace(
        kv_transfer_params={"kv_cache_control": {"mode": "pin", "cache_key": "k"}},
        request_id="r3",
        block_hashes=[b"h0", b"h1"],
    )


def _normal_request():
    return SimpleNamespace(kv_transfer_params=None, request_id="r2")


@pytest.fixture()
def patched_module(monkeypatch):
    """Inject fake vllm modules, import a fresh patch module, restore after.

    The module is loaded via spec_from_file_location so that the
    ``vllm_ascend.patch.platform`` package __init__ (which pulls in all
    sibling patch modules and needs a real vllm) is not executed.
    """
    _calls["allocate"].clear()
    _calls["cache_blocks"].clear()
    _calls["free"].clear()

    saved = {name: sys.modules.get(name) for name in _FAKE_MODULES}

    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_mod = types.ModuleType("vllm.v1.core")
    mgr_mod = types.ModuleType(_MGR_MODULE)
    mgr_mod.KVCacheManager = _FakeKVCacheManager
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.logger = logging.getLogger("vllm")
    logging_utils_mod = types.ModuleType("vllm.logging_utils")
    logging_utils_mod.ColoredFormatter = MagicMock()
    logging_utils_mod.NewLineFormatter = MagicMock()
    patch_pkg = types.ModuleType("vllm_ascend.patch")
    patch_pkg.__path__ = []
    platform_pkg = types.ModuleType("vllm_ascend.patch.platform")
    platform_pkg.__path__ = [str(_PATCH_FILE.parent)]
    vllm_mod.logger = logger_mod
    vllm_mod.v1 = v1_mod
    v1_mod.core = core_mod
    fake_map = dict(
        zip(
            _FAKE_MODULES,
            (
                vllm_mod,
                v1_mod,
                core_mod,
                mgr_mod,
                logger_mod,
                logging_utils_mod,
                patch_pkg,
                platform_pkg,
            ),
        )
    )
    for name, mod in fake_map.items():
        sys.modules[name] = mod

    try:
        spec = spec_from_file_location(_PATCH_MODULE, _PATCH_FILE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PATCH_MODULE] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop(_PATCH_MODULE, None)


def _make_manager():
    manager = _FakeKVCacheManager()
    assert isinstance(manager.kv_cache_control_manager, KVCacheControlManager)
    return manager


class TestNoStorePatch:
    def test_init_attaches_manager(self, patched_module):
        manager = _make_manager()
        assert manager.kv_cache_control_manager.metrics["no_store_requests"] == 0

    def test_allocate_forces_delay_for_no_store(self, patched_module):
        manager = _make_manager()
        result = manager.allocate_slots(_no_store_request(), 10)
        assert result == "ALLOCATED"
        assert len(_calls["allocate"]) == 1
        assert _calls["allocate"][0][1]["delay_cache_blocks"] is True

    def test_allocate_passthrough_without_declaration(self, patched_module):
        manager = _make_manager()
        manager.allocate_slots(_normal_request(), 10)
        assert "delay_cache_blocks" not in _calls["allocate"][0][1]

    def test_allocate_positional_guard(self, patched_module):
        manager = _make_manager()
        manager.allocate_slots(_no_store_request(), 10, 0, None, 0, 0, False)
        args, kwargs = _calls["allocate"][0]
        assert args == (0, None, 0, 0, True)
        assert "delay_cache_blocks" not in kwargs

    def test_allocate_keeps_caller_delay_true(self, patched_module):
        manager = _make_manager()
        manager.allocate_slots(_no_store_request(), 10, delay_cache_blocks=True)
        assert _calls["allocate"][0][1]["delay_cache_blocks"] is True

    def test_cache_blocks_noop_for_no_store(self, patched_module):
        manager = _make_manager()
        manager.cache_blocks(_no_store_request(), 128)
        assert _calls["cache_blocks"] == []

    def test_cache_blocks_passthrough_normal(self, patched_module):
        manager = _make_manager()
        manager.cache_blocks(_normal_request(), 128)
        assert _calls["cache_blocks"] == [128]

    def test_free_invokes_lifecycle_hook(self, patched_module):
        manager = _make_manager()
        manager.free(_pin_request())
        assert _calls["free"] == ["r3"]
        assert len(manager.kv_cache_control_manager._entries) == 1
        assert manager.kv_cache_control_manager.protection_level(b"h0") == 1

    def test_free_without_declaration_noop(self, patched_module):
        manager = _make_manager()
        manager.free(_normal_request())
        assert _calls["free"] == ["r2"]
        assert manager.kv_cache_control_manager._entries == {}

    def test_patch_is_idempotent(self, patched_module):
        first = _FakeKVCacheManager.allocate_slots
        patched_module._apply_patch()
        patched_module._apply_patch()
        assert _FakeKVCacheManager.allocate_slots is first
        assert getattr(first, _MARKER, False) is True

    def test_env_disable_skips_patch(self, patched_module, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_CONTROL", "0")
        sys.modules[_MGR_MODULE].KVCacheManager = type(
            "_FreshKVCacheManager", (), {"allocate_slots": lambda self, request, *a, **k: None}
        )
        fresh = importlib.reload(patched_module)
        assert fresh is not None
        assert getattr(sys.modules[_MGR_MODULE].KVCacheManager.allocate_slots, _MARKER, False) is False
