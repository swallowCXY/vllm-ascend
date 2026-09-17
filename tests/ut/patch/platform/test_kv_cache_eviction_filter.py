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

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_control_manager import (
    KVCacheControlManager,
    PinProtectedExhaustedError,
)

_PATCH_MODULE = "vllm_ascend.patch.platform.patch_kv_cache_eviction"
_PATCH_FILE = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_eviction.py"
_FAKE_MODULES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.block_pool",
    "vllm.v1.core.kv_cache_utils",
    "vllm.logger",
    "vllm_ascend.patch",
    "vllm_ascend.patch.platform",
)


class _FakeQueue:
    def __init__(self, blocks):
        self.blocks = list(blocks)

    def popleft(self):
        return self.blocks.pop(0)

    def popleft_n(self, n):
        out = self.blocks[:n]
        del self.blocks[:n]
        return out

    def append_n(self, blocks):
        self.blocks.extend(blocks)


class _FakeBlockPool:
    enable_caching = True
    metrics_collector = None

    def __init__(self, blocks, kvcm=None):
        self.free_block_queue = _FakeQueue(blocks)
        self.kv_cache_control_manager = kvcm

    def get_num_free_blocks(self):
        return len(self.free_block_queue.blocks)

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError("Cannot get free blocks from the pool")
        out = self.free_block_queue.popleft_n(num_blocks)
        for block in out:
            self._maybe_evict_cached_block(block)
            block.ref_cnt += 1
        return out

    def _maybe_evict_cached_block(self, block):
        block.evicted = True


def _block(block_id, block_hash=None, protected=False):
    return SimpleNamespace(
        block_id=block_id,
        block_hash=block_hash,
        ref_cnt=0,
        is_null=False,
        evicted=False,
    )


def _kvcm_with_entry(request_id, num_hashes):
    kvcm = KVCacheControlManager()
    req = SimpleNamespace(
        kv_transfer_params={"kv_cache_control": {"mode": "pin"}},
        request_id=request_id,
        block_hashes=[f"h{i}".encode() for i in range(num_hashes)],
    )
    kvcm.on_request_finished(req)
    return kvcm


@pytest.fixture()
def patched_module():
    saved = {name: sys.modules.get(name) for name in _FAKE_MODULES}
    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_mod = types.ModuleType("vllm.v1.core")
    block_pool_mod = types.ModuleType("vllm.v1.core.block_pool")
    block_pool_mod.BlockPool = _FakeBlockPool
    kv_cache_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils_mod.get_block_hash = lambda block_hash: block_hash
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.logger = logging.getLogger("vllm")
    patch_pkg = types.ModuleType("vllm_ascend.patch")
    patch_pkg.__path__ = []
    platform_pkg = types.ModuleType("vllm_ascend.patch.platform")
    platform_pkg.__path__ = [str(_PATCH_FILE.parent)]
    fake_map = dict(
        zip(
            _FAKE_MODULES,
            (
                vllm_mod,
                v1_mod,
                core_mod,
                block_pool_mod,
                kv_cache_utils_mod,
                logger_mod,
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


class TestEvictionFilter:
    def test_fast_path_without_protection(self, patched_module):
        blocks = [_block(i, f"h{i}".encode()) for i in range(3)]
        pool = _FakeBlockPool(blocks)
        out = pool.get_new_blocks(2)
        assert [b.block_id for b in out] == [0, 1]
        assert pool.free_block_queue.blocks[0].block_id == 2
        assert all(b.evicted for b in out)

    def test_protected_block_skipped_and_requeued(self, patched_module):
        kvcm = _kvcm_with_entry("k", 1)
        protected = _block(0, b"h0")
        blocks = [protected, _block(1, b"u1"), _block(2, b"u2")]
        pool = _FakeBlockPool(blocks, kvcm)
        out = pool.get_new_blocks(2)
        assert [b.block_id for b in out] == [1, 2]
        assert [b.block_id for b in pool.free_block_queue.blocks] == [0]

    def test_anonymous_blocks_unprotected(self, patched_module):
        kvcm = _kvcm_with_entry("k", 1)
        blocks = [_block(0), _block(1), _block(2)]
        pool = _FakeBlockPool(blocks, kvcm)
        out = pool.get_new_blocks(3)
        assert [b.block_id for b in out] == [0, 1, 2]

    def test_all_protected_raises_and_restores_queue(self, patched_module):
        kvcm = _kvcm_with_entry("k", 2)
        blocks = [_block(0, b"h0"), _block(1, b"h1")]
        pool = _FakeBlockPool(blocks, kvcm)
        with pytest.raises(PinProtectedExhaustedError):
            pool.get_new_blocks(2)
        assert [b.block_id for b in pool.free_block_queue.blocks] == [0, 1]
        assert all(not b.evicted for b in pool.free_block_queue.blocks)

    def test_partial_protection_exhaustion_raises_and_restores(self, patched_module):
        kvcm = _kvcm_with_entry("k", 2)
        blocks = [_block(0, b"h0"), _block(1, b"h1"), _block(2, b"u2")]
        pool = _FakeBlockPool(blocks, kvcm)
        with pytest.raises(PinProtectedExhaustedError):
            pool.get_new_blocks(3)
        assert [b.block_id for b in pool.free_block_queue.blocks] == [0, 1, 2]

    def test_insufficient_free_raises(self, patched_module):
        blocks = [_block(0, b"h0")]
        pool = _FakeBlockPool(blocks)
        with pytest.raises(ValueError):
            pool.get_new_blocks(3)
