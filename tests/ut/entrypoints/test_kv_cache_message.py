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
from vllm_ascend.entrypoints import kv_cache_message as kcm


def _msg(role="user", content="c", control=None):
    message = {"role": role, "content": content}
    if control is not None:
        message["kv_cache_control"] = control
    return message


class _FakeTokenizer:
    """Prefix-monotonic fake: rendered string = concatenation of contents."""

    def __init__(self, monotonic=True):
        self.monotonic = monotonic
        self.tokens_per_char = 0.25

    @staticmethod
    def _content(message):
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        text = "".join(self._content(message) for message in conversation)
        if not self.monotonic:
            # total-count marker breaks the prefix relation between renders
            text = f"[{len(conversation)}]" + text
        if not tokenize:
            return text
        return [0] * int(len(text) * self.tokens_per_char)


def _serving(monotonic=True):
    return SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=_FakeTokenizer(monotonic)),
        chat_template=None,
        default_chat_template_kwargs={},
    )


def _request(messages, tools=None):
    return SimpleNamespace(messages=messages, tools=tools, kv_transfer_params=None)


class TestAggregation:
    def test_pin_single_message(self):
        serving = _serving()
        request = _request([_msg("system", "sys"), _msg("user", "x" * 40, {"mode": "pin"})])
        directive = kcm.aggregate_message_controls(serving, request)
        assert directive["mode"] == "pin"
        # prefix covers system(3 chars) + pin message(40 chars) -> 43 chars * 0.25
        assert directive["pin_boundary_tokens"] == int(43 * 0.25)
        assert directive["ttl_s"] is None
        status = getattr(request, kcm._KVCC_STATUS_ATTR)
        assert status == {"status": "accepted", "reason": "pin"}

    def test_pin_boundary_includes_tools(self):
        serving = _serving()
        request = _request([_msg("user", "x" * 40, {"mode": "pin"})], tools=[{"name": "t"}])
        assert kcm.aggregate_message_controls(serving, request)["mode"] == "pin"

    def test_no_store_any_message_request_wide(self):
        serving = _serving()
        request = _request([_msg("user", "a"), _msg("user", "b", {"mode": "no_store"})])
        directive = kcm.aggregate_message_controls(serving, request)
        assert directive == {"mode": "no_store"}

    def test_release_any_message_request_wide(self):
        serving = _serving()
        request = _request([_msg("user", "a", {"mode": "release"}), _msg("user", "b")])
        assert kcm.aggregate_message_controls(serving, request) == {"mode": "release"}

    def test_conflicting_modes_rejected(self):
        serving = _serving()
        request = _request(
            [
                _msg("user", "a", {"mode": "pin"}),
                _msg("user", "b", {"mode": "no_store"}),
            ]
        )
        assert kcm.aggregate_message_controls(serving, request) is None
        status = getattr(request, kcm._KVCC_STATUS_ATTR)
        assert status["status"] == "rejected"
        assert status["reason"] == "conflict_modes"

    def test_release_no_store_conflict_rejected(self):
        serving = _serving()
        request = _request(
            [
                _msg("user", "a", {"mode": "release"}),
                _msg("user", "b", {"mode": "no_store"}),
            ]
        )
        assert kcm.aggregate_message_controls(serving, request) is None
        assert getattr(request, kcm._KVCC_STATUS_ATTR)["reason"] == "conflict_modes"

    def test_multiple_pin_messages_rejected(self):
        serving = _serving()
        request = _request(
            [
                _msg("user", "a", {"mode": "pin"}),
                _msg("user", "b", {"mode": "pin"}),
            ]
        )
        assert kcm.aggregate_message_controls(serving, request) is None
        assert getattr(request, kcm._KVCC_STATUS_ATTR)["reason"] == "multiple_pin_messages"

    def test_invalid_mode_ignored_when_alone(self):
        serving = _serving()
        request = _request([_msg("user", "a", {"mode": "weird"})])
        assert kcm.aggregate_message_controls(serving, request) is None
        assert getattr(request, kcm._KVCC_STATUS_ATTR)["reason"] == "invalid_declaration"

    def test_no_declarations_returns_none(self):
        serving = _serving()
        request = _request([_msg("user", "a"), _msg("user", "b")])
        assert kcm.aggregate_message_controls(serving, request) is None
        assert getattr(request, kcm._KVCC_STATUS_ATTR, None) is None

    def test_non_monotonic_template_rejected(self):
        serving = _serving(monotonic=False)
        request = _request(
            [
                _msg("user", "a"),
                _msg("user", "b" * 20, {"mode": "pin"}),
                _msg("user", "c"),
            ]
        )
        assert kcm.aggregate_message_controls(serving, request) is None
        assert getattr(request, kcm._KVCC_STATUS_ATTR)["reason"] == "boundary_render_failed"

    def test_pydantic_style_message(self):
        serving = _serving()
        message = SimpleNamespace(role="user", content="x" * 40, kv_cache_control={"mode": "pin"})
        request = _request([message])
        directive = kcm.aggregate_message_controls(serving, request)
        assert directive["mode"] == "pin"

    def test_batched_conversations_out_of_scope(self):
        serving = _serving()
        request = SimpleNamespace(messages=[[_msg("user", "a", {"mode": "pin"})]], tools=None)
        assert kcm.aggregate_message_controls(serving, request) is None


class TestInjection:
    def test_inject_merges_into_existing_params(self):
        serving = _serving()
        request = _request([_msg("user", "x" * 40, {"mode": "pin"})])
        request.kv_transfer_params = {"kv_transfer_params": {"do_remote_prefill": True}}
        kcm.inject_request_control(serving, request)
        injected = request.kv_transfer_params["kv_cache_control"]
        assert injected["mode"] == "pin"
        assert request.kv_transfer_params["kv_transfer_params"] == {"do_remote_prefill": True}

    def test_inject_noop_when_no_declaration(self):
        serving = _serving()
        request = _request([_msg("user", "a")])
        kcm.inject_request_control(serving, request)
        assert request.kv_transfer_params is None

    def test_pin_ttl_passes_through(self):
        serving = _serving()
        request = _request([_msg("user", "x" * 40, {"mode": "pin", "ttl_s": 90})])
        kcm.inject_request_control(serving, request)
        assert request.kv_transfer_params["kv_cache_control"]["ttl_s"] == 90


class TestResponseStatus:
    def test_attach_to_response(self):
        request = SimpleNamespace(**{kcm._KVCC_STATUS_ATTR: {"status": "accepted", "reason": "pin"}})
        response = SimpleNamespace()
        kcm.attach_status_to_response(request, response)
        assert response.kv_cache_control_status == {"status": "accepted", "reason": "pin"}

    def test_attach_skipped_without_status(self):
        request = SimpleNamespace()
        response = SimpleNamespace()
        kcm.attach_status_to_response(request, response)
        assert not hasattr(response, "kv_cache_control_status")

    def test_attach_survives_readonly_response(self):
        request = SimpleNamespace(**{kcm._KVCC_STATUS_ATTR: {"status": "rejected", "reason": "conflict_modes"}})

        class _Frozen:
            __slots__ = ()

        kcm.attach_status_to_response(request, _Frozen())
