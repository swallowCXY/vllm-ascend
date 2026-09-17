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
"""HTTP control-plane route for KV cache release.

Mounted by ``patch_kv_cache_control_engine`` via ``build_app``. This is an
operational control endpoint (same trust level as ``/reset_prefix_cache``);
authentication is expected to be handled by the deployment gateway.
"""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/kv_cache")


class ReleaseRequest(BaseModel):
    request_id: str


async def _invoke(request: Request, method: str, *args) -> Any:
    client = request.app.state.engine_client
    call_async = getattr(client, "call_utility_async", None)
    if call_async is not None:
        return await call_async(method, *args)
    return client.call_utility(method, *args)


@router.post("/release")
async def release(req: ReleaseRequest, raw: Request):
    released = await _invoke(raw, "kv_cache_release", req.request_id)
    return {"released": released}


def attach_router(app) -> None:
    app.include_router(router)
