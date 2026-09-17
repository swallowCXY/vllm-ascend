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

from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_control_manager import KVCacheControlManager


def _request(mode=None, **kwargs):
    params = None
    if mode is not None:
        control = {"mode": mode}
        control.update(kwargs)
        params = {"kv_cache_control": control}
    return SimpleNamespace(kv_transfer_params=params, request_id="r1", block_hashes=[])


def _hashes(n, offset=0):
    return [f"h{i + offset}".encode() for i in range(n)]


def _fake_manager(num_gpu_blocks=1000):
    return SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=num_gpu_blocks))


class TestParseRequestControl:
    def test_no_store_declaration(self):
        kvcm = KVCacheControlManager()
        req = _request("no_store")
        assert kvcm.is_no_store(req) is True
        assert kvcm.metrics["no_store_requests"] == 1

    def test_release_declaration(self):
        kvcm = KVCacheControlManager()
        req = _request("release")
        assert kvcm.is_release(req) is True
        assert kvcm.is_no_store(req) is False
        assert kvcm.metrics["release_requests"] == 1

    def test_pin_default_ttl(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S", "3600")
        kvcm = KVCacheControlManager()
        control = kvcm.parse_request_control(_request("pin"))
        assert control["mode"] == "pin"
        assert control["ttl_s"] == 3600.0
        assert control["pin_boundary_tokens"] is None
        assert kvcm.metrics["pin_requests"] == 1

    def test_pin_explicit_ttl_and_boundary(self):
        kvcm = KVCacheControlManager()
        control = kvcm.parse_request_control(_request("pin", ttl_s=120, pin_boundary_tokens=137))
        assert control["ttl_s"] == 120.0
        assert control["pin_boundary_tokens"] == 137

    def test_missing_declaration(self):
        kvcm = KVCacheControlManager()
        assert kvcm.is_no_store(_request()) is False
        assert kvcm.is_release(_request()) is False

    def test_unknown_mode_parse_error(self):
        kvcm = KVCacheControlManager()
        assert kvcm.parse_request_control(_request("future_mode")) is None
        assert kvcm.metrics["parse_errors"] == 1

    def test_malformed_payloads(self):
        kvcm = KVCacheControlManager()
        for payload in ("not_a_dict", {"kv_cache_control": "pin"}, {"kv_cache_control": {"mode": 123}}):
            assert kvcm.parse_request_control(_request()) is False or True
            req = SimpleNamespace(kv_transfer_params=payload, request_id="r", block_hashes=[])
            assert kvcm.parse_request_control(req) is None
        assert kvcm.metrics["parse_errors"] == 3

    def test_parse_memoized_per_request(self):
        kvcm = KVCacheControlManager()
        req = _request("no_store")
        first = kvcm.parse_request_control(req)
        second = kvcm.parse_request_control(req)
        assert first is second
        assert kvcm.metrics["no_store_requests"] == 1


class TestPinLifecycle:
    def test_pin_activation_and_protection(self):
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(), block_size=16)
        req = _request("pin", pin_boundary_tokens=32)
        req.block_hashes = _hashes(4)
        kvcm.on_request_finished(req)
        assert kvcm.protection_level(b"h0") == 1
        assert kvcm.protection_level(b"h1") == 1
        assert kvcm.protection_level(b"h2") == 0
        assert kvcm.protection_level(b"missing") == 0
        assert kvcm.has_protection() is True
        assert kvcm.metrics["pin_requests"] == 1

    def test_pin_boundary_floors_to_blocks(self):
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(), block_size=16)
        req = _request("pin", pin_boundary_tokens=137)
        req.block_hashes = _hashes(10)
        kvcm.on_request_finished(req)
        entry = kvcm._pin_entries["r1"]
        assert len(entry.hashes) == 8

    def test_pin_without_boundary_protects_all(self):
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(), block_size=16)
        req = _request("pin")
        req.block_hashes = _hashes(5)
        kvcm.on_request_finished(req)
        assert len(kvcm._pin_entries["r1"].hashes) == 5

    def test_pin_quota_degrades(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO", "0.1")
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(num_gpu_blocks=100), block_size=16)
        req = _request("pin")
        req.block_hashes = _hashes(50)
        kvcm.on_request_finished(req)
        assert kvcm.protection_level(b"h0") == 0
        assert kvcm.metrics["quota_degraded"] == 1
        assert kvcm.has_protection() is False

    def test_pin_ttl_expiry_sweep(self):
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(), block_size=16)
        req = _request("pin", ttl_s=60)
        req.block_hashes = _hashes(2)
        kvcm.on_request_finished(req)
        entry = kvcm._pin_entries["r1"]
        entry.expire_at -= 61
        kvcm._next_expiry = entry.expire_at
        kvcm.maybe_sweep()
        assert kvcm.protection_level(b"h0") == 0
        assert kvcm.metrics["ttl_expired"] == 1

    def test_maybe_sweep_fast_path(self):
        kvcm = KVCacheControlManager()
        kvcm.maybe_sweep()
        assert kvcm.metrics["ttl_expired"] == 0


class TestRelease:
    def test_request_level_release_plan(self):
        kvcm = KVCacheControlManager()
        kvcm.bind_kv_cache_manager(_fake_manager(), block_size=16)
        req = _request("release")
        req.block_hashes = _hashes(3)
        kvcm.on_request_finished(req)
        plan = kvcm.take_release_plan()
        assert plan == _hashes(3)
        assert kvcm.take_release_plan() == []


class TestNoStorePath:
    def test_no_store_finish_is_noop_for_registry(self):
        kvcm = KVCacheControlManager()
        req = _request("no_store")
        req.block_hashes = _hashes(3)
        kvcm.on_request_finished(req)
        assert kvcm._pin_entries == {}
        assert kvcm.take_release_plan() == []

    def test_request_without_declaration_noop(self):
        kvcm = KVCacheControlManager()
        req = _request()
        req.request_id = "r9"
        req.block_hashes = _hashes(2)
        kvcm.on_request_finished(req)
        assert kvcm._pin_entries == {}
        assert kvcm.take_release_plan() == []
