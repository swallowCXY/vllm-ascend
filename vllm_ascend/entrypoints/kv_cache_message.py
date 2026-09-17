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
"""Message-level ``kv_cache_control`` extraction and boundary computation.

Agents declare lifecycle modes on individual chat messages::

    {"role": "user", "content": "...", "kv_cache_control": {"mode": "pin"}}

Rules (mutually exclusive per request):
- ``pin``: at most one message; protects tools + all messages up to and
  including the declared one (block-aligned tail truncation); hard
  protection until TTL expiry (default ``VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S``).
- ``no_store`` / ``release``: declared on any message, effective for the
  whole request.

The serving layer aggregates the declarations into the request-level
``kv_transfer_params`` carrier consumed by the engine; message boundaries are
computed by incremental chat-template rendering with a prefix-monotonicity
check. Rejected declarations never fail the request; the outcome is reported
in the response field ``kv_cache_control_status`` (non-streaming) and logs.
"""

import contextlib
from typing import Any

from vllm.logger import logger

_MODE_PIN = "pin"
_MODE_NO_STORE = "no_store"
_MODE_RELEASE = "release"
_SUPPORTED_MODES = (_MODE_PIN, _MODE_NO_STORE, _MODE_RELEASE)
_KVCC_STATUS_ATTR = "_kvcc_status"


def _get_message_control(message: Any) -> dict[str, Any] | None:
    """Read ``kv_cache_control`` from a dict or pydantic message object."""
    control = None
    if isinstance(message, dict):
        control = message.get("kv_cache_control")
    else:
        extra = getattr(message, "model_extra", None) or {}
        control = extra.get("kv_cache_control")
        if control is None:
            control = getattr(message, "kv_cache_control", None)
    if control is None:
        return None
    if not isinstance(control, dict) or control.get("mode") not in _SUPPORTED_MODES:
        return {"mode": "__invalid__"}
    return control


def _messages_of(request: Any) -> list[Any] | None:
    messages = getattr(request, "messages", None)
    if not isinstance(messages, list):
        # batched conversations and other shapes are out of scope
        return None
    return messages


def aggregate_message_controls(serving: Any, request: Any) -> dict[str, Any] | None:
    """Extract and adjudicate per-message declarations on ``request``.

    Returns the aggregated engine directive (already validated), or ``None``
    when nothing (or a rejected mix) was declared. The adjudication outcome
    is stashed on the request for the response status field.
    """
    messages = _messages_of(request)
    if not messages:
        return None
    declared: list[tuple[int, dict[str, Any]]] = []
    invalid = False
    for index, message in enumerate(messages):
        control = _get_message_control(message)
        if control is None:
            continue
        if control.get("mode") == "__invalid__":
            invalid = True
            continue
        declared.append((index, control))

    if invalid:
        _stash_status(request, "rejected", "invalid_declaration")
        return None
    if not declared:
        return None

    modes = {control["mode"] for _, control in declared}
    if len(modes) > 1:
        _stash_status(request, "rejected", "conflict_modes", sorted(modes))
        return None
    mode = modes.pop()

    if mode == _MODE_PIN:
        pin_declared = [(index, control) for index, control in declared if control["mode"] == _MODE_PIN]
        if len(pin_declared) > 1:
            _stash_status(request, "rejected", "multiple_pin_messages")
            return None
        index, control = pin_declared[0]
        boundary_tokens = _pin_boundary_tokens(serving, request, index)
        if boundary_tokens is None:
            _stash_status(request, "rejected", "boundary_render_failed")
            return None
        _stash_status(request, "accepted", "pin")
        return {
            "mode": _MODE_PIN,
            "ttl_s": control.get("ttl_s"),
            "pin_boundary_tokens": boundary_tokens,
        }
    _stash_status(request, "accepted", mode)
    return {"mode": mode}


def _stash_status(request: Any, status: str, reason: str, detail: Any = None) -> None:
    payload: dict[str, Any] = {"status": status, "reason": reason}
    if detail is not None:
        payload["detail"] = detail
    with contextlib.suppress(Exception):
        setattr(request, _KVCC_STATUS_ATTR, payload)
    logger.info("KV cache control declaration %s: %s", status, payload)


def _render_conversation(serving: Any, request: Any, conversation: list[Any], tokenize: bool) -> Any:
    """Render a conversation prefix with the serving chat template + tools."""
    tokenizer = serving.renderer.tokenizer
    if tokenizer is None:
        raise RuntimeError("tokenizer unavailable")
    kwargs: dict[str, Any] = {}
    if getattr(serving, "chat_template", None):
        kwargs["chat_template"] = serving.chat_template
    tools = getattr(request, "tools", None)
    if tools:
        kwargs["tools"] = tools
    merged_kwargs = dict(getattr(serving, "default_chat_template_kwargs", {}) or {})
    request_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    merged_kwargs.update(request_kwargs)
    kwargs.update(merged_kwargs)
    return tokenizer.apply_chat_template(conversation=conversation, tokenize=tokenize, **kwargs)


def _pin_boundary_tokens(serving: Any, request: Any, pin_index: int) -> int | None:
    """Token length of the rendered prefix up to and including the pin message.

    Uses incremental chat-template rendering with a prefix-monotonicity check
    (the rendered prefix must be a string prefix of the full rendering);
    returns ``None`` when the template is not prefix-monotonic or rendering
    fails.
    """
    try:
        messages = _messages_of(request)
        if messages is None:
            return None
        prefix_str = _render_conversation(serving, request, messages[: pin_index + 1], tokenize=False)
        if not isinstance(prefix_str, str):
            return None
        full_str = _render_conversation(serving, request, messages, tokenize=False)
        if not isinstance(full_str, str) or not full_str.startswith(prefix_str):
            logger.warning("Chat template is not prefix-monotonic; message-level pin rejected for request")
            return None
        prefix_ids = _render_conversation(serving, request, messages[: pin_index + 1], tokenize=True)
        if isinstance(prefix_ids, dict):
            prefix_ids = prefix_ids.get("input_ids")
        return len(prefix_ids)
    except Exception:
        logger.exception("Failed to compute message-level pin boundary")
        return None


def inject_request_control(serving: Any, request: Any) -> None:
    """Entry point: aggregate message declarations into ``kv_transfer_params``.

    Called from the ``OpenAIServingChat.render_chat_request`` wrapper, i.e.
    before the engine inputs are built.
    """
    try:
        directive = aggregate_message_controls(serving, request)
    except Exception:
        logger.exception("Failed to aggregate message-level kv_cache_control")
        return
    if directive is None:
        return
    params = dict(getattr(request, "kv_transfer_params", None) or {})
    params["kv_cache_control"] = directive
    try:
        request.kv_transfer_params = params
    except Exception:
        logger.exception("Failed to inject kv_cache_control into request")


def attach_status_to_response(request: Any, response: Any) -> None:
    """Attach the adjudication outcome to a non-streaming response."""
    status = getattr(request, _KVCC_STATUS_ATTR, None)
    if status is None or response is None:
        return
    try:
        response.kv_cache_control_status = status
    except Exception:
        logger.debug("Could not attach kv_cache_control_status to response")
