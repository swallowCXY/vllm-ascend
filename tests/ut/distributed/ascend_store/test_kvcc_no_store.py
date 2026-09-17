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
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_control_manager import KVCacheControlManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)


def _scheduler(request_flags: bool = True) -> KVPoolScheduler:
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.model_name = "test-model"
    scheduler.kv_cache_control = KVCacheControlManager() if request_flags else None
    scheduler._pending_delete_keys = []
    scheduler.kv_role = "kv_producer"
    scheduler.consumer_is_to_put = False
    scheduler.use_layerwise = False
    scheduler._delayed_free_req_ids = set()
    scheduler._request_trackers = {}
    scheduler._unfinished_requests = {}
    scheduler._unfinished_request_ids = set()
    scheduler._preempted_req_ids = set()
    scheduler._loading_req_ids = set()
    return scheduler


def _request(request_id, no_store):
    params = {"kv_cache_control": {"mode": "no_store"}} if no_store else None
    return SimpleNamespace(
        kv_transfer_params=params,
        request_id=request_id,
        req_id=request_id,
    )


class TestRequestFinishedNoStore:
    def test_no_store_request_returns_no_delay(self):
        scheduler = _scheduler()
        scheduler._request_trackers["r1"] = SimpleNamespace(num_saved_tokens=100)
        delay, meta = scheduler.request_finished(_request("r1", no_store=True), [1, 2])
        assert delay is False and meta is None
        assert "r1" not in scheduler._delayed_free_req_ids

    def test_normal_request_still_delays_free(self):
        scheduler = _scheduler()
        scheduler._request_trackers["r1"] = SimpleNamespace(num_saved_tokens=100)
        delay, _ = scheduler.request_finished(_request("r1", no_store=False), [1, 2])
        assert delay is True
        assert "r1" in scheduler._delayed_free_req_ids

    def test_no_store_without_manager(self):
        scheduler = _scheduler(request_flags=False)
        scheduler._request_trackers["r1"] = SimpleNamespace(num_saved_tokens=100)
        delay, _ = scheduler.request_finished(_request("r1", no_store=True), [1, 2])
        assert delay is True

    def test_all_groups_variant_no_store(self):
        scheduler = _scheduler()
        scheduler._request_trackers["r1"] = SimpleNamespace(num_saved_tokens=100)
        delay, meta = scheduler.request_finished_all_groups(_request("r1", no_store=True), ([1], [2]))
        assert delay is False and meta is None
        assert "r1" not in scheduler._delayed_free_req_ids


class TestBuildConnectorMetaNoStore:
    def _scheduler_with_output(self):
        scheduler = _scheduler()
        scheduler._process_new_request = MagicMock(return_value=None)
        scheduler._process_running_cached_request = MagicMock(return_value=None)
        scheduler._process_preempted_cached_request = MagicMock(return_value=None)
        scheduler.touch_sending_mamba_blocks = MagicMock()
        output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        )
        return scheduler, output

    def test_new_request_skipped(self):
        scheduler, output = self._scheduler_with_output()
        output.scheduled_new_reqs = [_request("r1", no_store=True)]
        scheduler.build_connector_meta(output)
        scheduler._process_new_request.assert_not_called()

    def test_new_request_processed_when_normal(self):
        scheduler, output = self._scheduler_with_output()
        output.scheduled_new_reqs = [_request("r1", no_store=False)]
        scheduler.build_connector_meta(output)
        scheduler._process_new_request.assert_called_once()

    def test_running_cached_request_skipped(self):
        scheduler, output = self._scheduler_with_output()
        req = _request("r1", no_store=True)
        scheduler._unfinished_requests["r1"] = (req, [[1]])
        output.scheduled_cached_reqs = SimpleNamespace(req_ids=["r1"], new_block_ids=[[1]])
        scheduler.build_connector_meta(output)
        scheduler._process_running_cached_request.assert_not_called()

    def test_running_cached_request_processed_when_normal(self):
        scheduler, output = self._scheduler_with_output()
        req = _request("r1", no_store=False)
        scheduler._unfinished_requests["r1"] = (req, [[1]])
        output.scheduled_cached_reqs = SimpleNamespace(req_ids=["r1"], new_block_ids=[[1]])
        scheduler.build_connector_meta(output)
        scheduler._process_running_cached_request.assert_called_once()

    def test_cached_request_without_tracker_processed(self):
        scheduler, output = self._scheduler_with_output()
        output.scheduled_cached_reqs = SimpleNamespace(req_ids=["unknown"], new_block_ids=[[1]])
        scheduler.build_connector_meta(output)
        scheduler._process_running_cached_request.assert_called_once()
