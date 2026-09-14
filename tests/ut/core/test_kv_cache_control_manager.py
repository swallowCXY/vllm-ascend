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

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_control_manager import (
    ContentRef,
    KVCacheControlManager,
    PinHandle,
)


def _request(kv_transfer_params=None):
    return SimpleNamespace(kv_transfer_params=kv_transfer_params, request_id="r1")


def _no_store_request():
    return _request({"kv_cache_control": {"mode": "no_store"}})


class TestParseRequestControl:
    def test_no_store_declaration(self):
        kvcm = KVCacheControlManager()
        req = _no_store_request()
        assert kvcm.is_no_store(req) is True
        assert kvcm.metrics["no_store_requests"] == 1

    def test_missing_declaration(self):
        kvcm = KVCacheControlManager()
        assert kvcm.is_no_store(_request(None)) is False
        assert kvcm.is_no_store(_request({})) is False
        assert kvcm.metrics["no_store_requests"] == 0

    def test_unknown_mode_ignored(self):
        kvcm = KVCacheControlManager()
        req = _request({"kv_cache_control": {"mode": "future_mode"}})
        assert kvcm.is_no_store(req) is False
        assert kvcm.metrics["unsupported_requests"] == 1

    def test_unsupported_modes_ignored(self):
        kvcm = KVCacheControlManager()
        for mode in ("pin", "ttl"):
            req = _request({"kv_cache_control": {"mode": mode, "ttl_s": 60}})
            assert kvcm.is_no_store(req) is False
        assert kvcm.metrics["unsupported_requests"] == 2

    @pytest.mark.parametrize(
        "payload",
        [
            "not_a_dict",
            {"kv_cache_control": "no_store"},
            {"kv_cache_control": {"mode": 123}},
            {"kv_cache_control": None},
        ],
    )
    def test_malformed_payloads(self, payload):
        kvcm = KVCacheControlManager()
        req = _request(payload)
        if payload == {"kv_cache_control": None}:
            assert kvcm.is_no_store(req) is False
            assert kvcm.metrics["parse_errors"] == 0
        else:
            assert kvcm.is_no_store(req) is False
            assert kvcm.metrics["parse_errors"] == 1

    def test_kv_transfer_params_wrong_type(self):
        kvcm = KVCacheControlManager()
        assert kvcm.is_no_store(_request("oops")) is False
        assert kvcm.metrics["parse_errors"] == 1

    def test_request_without_attribute(self):
        kvcm = KVCacheControlManager()
        assert kvcm.is_no_store(SimpleNamespace()) is False

    def test_parse_memoized_per_request(self):
        kvcm = KVCacheControlManager()
        req = _no_store_request()
        first = kvcm.parse_request_control(req)
        second = kvcm.parse_request_control(req)
        assert first is second
        assert kvcm.metrics["no_store_requests"] == 1


class TestInterfaceStubs:
    def test_stubs_raise_not_implemented(self):
        kvcm = KVCacheControlManager()
        ref = ContentRef(cache_key="kb:doc:1")
        with pytest.raises(NotImplementedError):
            kvcm.pin(ref, priority=1, ttl_s=60)
        with pytest.raises(NotImplementedError):
            kvcm.set_ttl(ref, 60)
        with pytest.raises(NotImplementedError):
            kvcm.release(ref)

    def test_content_ref_and_pin_handle(self):
        ref = ContentRef(namespace="ns", cache_key="k", prefix_tokens=137)
        assert ref.prefix_tokens == 137
        handle = PinHandle.generate()
        assert isinstance(handle, str) and len(handle) == 32
