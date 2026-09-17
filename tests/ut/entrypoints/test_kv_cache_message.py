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
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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


class TestChatServingPatch:
    """Wrappers installed by ``patch_kv_cache_control._patch_chat_serving``."""

    @pytest.fixture()
    def patched_serving(self):
        """Fake the chat-serving module and (re)load the control patch module.

        Restores ``sys.modules`` afterwards so other test files are unaffected.
        """
        fake_names = (
            "vllm",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.kv_cache_manager",
            "vllm.v1.core.kv_cache_utils",
            "vllm_ascend.patch",
            "vllm_ascend.patch.platform",
        )
        saved = {name: sys.modules.get(name) for name in fake_names}
        vllm_mod = types.ModuleType("vllm")
        v1_mod = types.ModuleType("vllm.v1")
        core_mod = types.ModuleType("vllm.v1.core")

        class _FakeKVCacheManager:
            def allocate_slots(self, *args, **kwargs):
                return None

            def cache_blocks(self, *args, **kwargs):
                return None

            def free(self, *args, **kwargs):
                return None

        mgr_mod = types.ModuleType("vllm.v1.core.kv_cache_manager")
        mgr_mod.KVCacheManager = _FakeKVCacheManager
        kv_cache_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
        kv_cache_utils_mod.get_block_hash = lambda block_hash: block_hash
        kv_cache_utils_mod.make_block_hash_with_group_id = lambda block_hash, group_id: (block_hash, group_id)

        class _FakeServingChat:
            def __init__(self):
                self.renderer = SimpleNamespace(tokenizer=_FakeTokenizer())
                self.chat_template = None
                self.default_chat_template_kwargs = {}

            async def render_chat_request(self, request):
                return "RENDERED"

            async def create_chat_completion(self, request, raw_request=None):
                return SimpleNamespace(mark="RESPONSE")

        serving_mod = types.ModuleType("vllm.entrypoints.openai.chat_completion.serving")
        serving_mod.OpenAIServingChat = _FakeServingChat
        openai_mod = types.ModuleType("vllm.entrypoints.openai")
        openai_mod.chat_completion = types.ModuleType("vllm.entrypoints.openai.chat_completion")
        openai_mod.chat_completion.serving = serving_mod
        vllm_mod.v1 = v1_mod
        v1_mod.core = core_mod
        openai_pkg = types.ModuleType("vllm.entrypoints.openai")
        openai_pkg.chat_completion = openai_mod.chat_completion
        entrypoints_mod = types.ModuleType("vllm.entrypoints")
        entrypoints_mod.openai = openai_pkg
        patch_pkg = types.ModuleType("vllm_ascend.patch")
        patch_pkg.__path__ = []
        platform_pkg = types.ModuleType("vllm_ascend.patch.platform")
        patch_file = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_kv_cache_control.py"
        platform_pkg.__path__ = [str(patch_file.parent)]
        for name, mod in (
            ("vllm", vllm_mod),
            ("vllm.v1", v1_mod),
            ("vllm.v1.core", core_mod),
            ("vllm.v1.core.kv_cache_manager", mgr_mod),
            ("vllm.v1.core.kv_cache_utils", kv_cache_utils_mod),
            ("vllm.entrypoints", entrypoints_mod),
            ("vllm.entrypoints.openai", openai_pkg),
            ("vllm.entrypoints.openai.chat_completion", openai_mod.chat_completion),
            ("vllm.entrypoints.openai.chat_completion.serving", serving_mod),
            ("vllm_ascend.patch", patch_pkg),
            ("vllm_ascend.patch.platform", platform_pkg),
        ):
            sys.modules[name] = mod
        try:
            spec = importlib.util.spec_from_file_location(
                "vllm_ascend.patch.platform.patch_kv_cache_control", patch_file
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            yield mod, _FakeServingChat
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            sys.modules.pop("vllm_ascend.patch.platform.patch_kv_cache_control", None)

    def test_wrappers_installed(self, patched_serving):
        mod, serving_cls = patched_serving
        assert getattr(serving_cls.render_chat_request, "__vcc_no_store_patched__", False)
        assert getattr(serving_cls.create_chat_completion, "__vcc_no_store_patched__", False)

    def test_render_wrapper_injects(self, patched_serving):
        _, serving_cls = patched_serving
        serving = serving_cls()
        request = _request([_msg("user", "a"), _msg("user", "b" * 40, {"mode": "pin"})])
        result = asyncio.run(serving.render_chat_request(request))
        assert result == "RENDERED"
        injected = request.kv_transfer_params["kv_cache_control"]
        assert injected["mode"] == "pin"
        assert injected["pin_boundary_tokens"] == int(41 * 0.25)

    def test_create_wrapper_attaches_status(self, patched_serving):
        _, serving_cls = patched_serving
        serving = serving_cls()
        request = _request([_msg("user", "a", {"mode": "pin"})])
        asyncio.run(serving.render_chat_request(request))
        raw = object()
        result = asyncio.run(serving.create_chat_completion(request, raw))
        assert result.mark == "RESPONSE"
        assert result.kv_cache_control_status == {"status": "accepted", "reason": "pin"}

    def test_create_wrapper_no_status_for_undeclared(self, patched_serving):
        _, serving_cls = patched_serving
        serving = serving_cls()
        request = _request([_msg("user", "a")])
        result = asyncio.run(serving.create_chat_completion(request, None))
        assert not hasattr(result, "kv_cache_control_status")

    def test_patch_is_idempotent(self, patched_serving):
        mod, serving_cls = patched_serving
        first = serving_cls.render_chat_request
        mod._apply_patch()
        assert serving_cls.render_chat_request is first
