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
"""Eviction-time protection filter for pinned / TTL KV cache entries.

Wraps ``BlockPool.get_new_blocks``: when the free queue front holds blocks
protected by ``KVCacheControlManager`` entries (pin / unexpired TTL), those
blocks are skipped and re-appended to the queue tail (LRU refresh), and
unprotected blocks are taken instead. Soft-pin invariant: if the whole queue
is protected, the coldest protected blocks (queue order) are sacrificed so
that allocation is never blocked by protection.
"""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_block_hash

from vllm_ascend import envs

_PATCH_MARKER = "__vcc_eviction_filter_patched__"


def _apply_patch() -> None:
    if not envs.VLLM_ASCEND_KV_CACHE_CONTROL:
        return
    if getattr(BlockPool.get_new_blocks, _PATCH_MARKER, False):
        return

    original = BlockPool.get_new_blocks

    @wraps(original)
    def _patched_get_new_blocks(self: BlockPool, num_blocks: int) -> list[Any]:
        kvcm = getattr(self, "kv_cache_control_manager", None)
        if kvcm is None or not kvcm.has_protection() or num_blocks > self.get_num_free_blocks():
            return original(self, num_blocks)

        collected: list[Any] = []
        skipped: list[Any] = []
        queue = self.free_block_queue
        remaining = self.get_num_free_blocks()
        while len(collected) < num_blocks and remaining > 0:
            block = queue.popleft()
            remaining -= 1
            level = 0
            if block.block_hash is not None:
                level = kvcm.protection_level(get_block_hash(block.block_hash))
            if level > 0:
                skipped.append(block)
            else:
                collected.append(block)
        if len(collected) < num_blocks:
            # Soft-pin fallback: sacrifice the coldest protected blocks (queue
            # order) so that allocation is never blocked by protection.
            fallback = num_blocks - len(collected)
            collected.extend(skipped[:fallback])
            skipped = skipped[fallback:]
            kvcm.metrics["pin_evictions"] += fallback
            logger.debug(
                "KV cache protection fallback: evicted %d protected blocks under pressure",
                fallback,
            )
        if skipped:
            queue.append_n(skipped)

        for block in collected:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return collected

    _patched_get_new_blocks.__vcc_eviction_filter_patched__ = True  # type: ignore[attr-defined]
    BlockPool.get_new_blocks = _patched_get_new_blocks
    logger.info("KV cache control enabled: eviction protection filter applied")


_apply_patch()
