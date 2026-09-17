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
"""Block-level release helper shared by the KV cache control patches."""

from logging import getLogger
from typing import Any

from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

logger = getLogger(__name__)


def evict_hashes(manager: Any, hashes: list[Any]) -> int:
    """Unregister prefix-cache registrations of raw block hashes (all groups).

    ``manager`` is a ``KVCacheManager``; only hash registrations are removed
    (upstream ``evict_blocks``), blocks stay allocatable.
    """
    if not hashes:
        return 0
    block_pool = manager.block_pool
    num_groups = len(manager.kv_cache_config.kv_cache_groups)
    block_ids: set[int] = set()
    for block_hash in hashes:
        for group_id in range(num_groups):
            block = block_pool.cached_block_hash_to_block.get_one_block(
                make_block_hash_with_group_id(block_hash, group_id)
            )
            if block is not None:
                block_ids.add(block.block_id)
    if block_ids:
        block_pool.evict_blocks(block_ids)
    return len(block_ids)
