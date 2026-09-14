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
"""HTTP control-plane routes for KV cache lifecycle management.

Mounted by ``patch_kv_cache_control_engine`` via ``build_app``. These are
operational control endpoints (same trust level as ``/reset_prefix_cache``);
authentication is expected to be handled by the deployment gateway.
"""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/kv_cache")


async def _invoke(request: Request, method: str, *args: Any) -> Any:
    client = request.app.state.engine_client
    call_async = getattr(client, "call_utility_async", None)
    if call_async is not None:
        return await call_async(method, *args)
    return client.call_utility(method, *args)


class PinRequest(BaseModel):
    cache_key: str
    namespace: str = "default"
    priority: int = 0
    ttl_s: float | None = None
    tier: str = "hbm"


class TtlRequest(BaseModel):
    cache_key: str
    namespace: str = "default"
    ttl_s: float | None = None


class KeyRequest(BaseModel):
    cache_key: str
    namespace: str = "default"


class FlushRequest(BaseModel):
    keep_protected: bool = True


@router.post("/pin")
async def pin(req: PinRequest, raw: Request):
    handle = await _invoke(raw, "kv_cache_pin", req.namespace, req.cache_key, req.priority, req.ttl_s, req.tier)
    return {"handle": handle}


@router.post("/ttl")
async def set_ttl(req: TtlRequest, raw: Request):
    await _invoke(raw, "kv_cache_set_ttl", req.namespace, req.cache_key, req.ttl_s)
    return {"ok": True}


@router.post("/release")
async def release(req: KeyRequest, raw: Request):
    released = await _invoke(raw, "kv_cache_release", req.namespace, req.cache_key)
    return {"released": released}


@router.post("/flush")
async def flush(req: FlushRequest, raw: Request):
    evicted = await _invoke(raw, "kv_cache_flush", req.keep_protected)
    return {"evicted_blocks": evicted}


def attach_router(app) -> None:
    app.include_router(router)
