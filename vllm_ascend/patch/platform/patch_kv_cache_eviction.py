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
"""Hard eviction protection for pinned KV cache entries.

Wraps ``BlockPool.get_new_blocks``: blocks protected by an unexpired pin are
never handed out as victims. When the free queue cannot satisfy the request
without touching protected blocks, the skipped blocks are re-queued and
``PinProtectedExhaustedError`` is raised; the ``allocate_slots`` wrapper in
``patch_kv_cache_control`` converts it into an allocation failure (``None``)
so the scheduler takes its normal preemption path. Hard pins are therefore
never evicted, at the cost of reduced capacity for normal traffic once the
pin budget is consumed.
"""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_block_hash

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_control_manager import PinProtectedExhaustedError

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
        if kvcm is None or not kvcm.has_protection():
            return original(self, num_blocks)

        queue = self.free_block_queue
        remaining = self.get_num_free_blocks()
        popped: list[tuple[Any, bool]] = []
        unprotected_count = 0
        while unprotected_count < num_blocks and remaining > 0:
            block = queue.popleft()
            remaining -= 1
            level = 0
            if block.block_hash is not None:
                level = kvcm.protection_level(get_block_hash(block.block_hash))
            protected = level > 0
            popped.append((block, protected))
            if not protected:
                unprotected_count += 1

        collected = [block for block, protected in popped if not protected]
        if len(collected) < num_blocks:
            # Hard protection: never hand out protected blocks. Restore the
            # queue in its original order and fail this allocation; the
            # allocate_slots wrapper maps this to None so the scheduler
            # preempts normally.
            queue.append_n([block for block, _ in popped])
            logger.debug(
                "KV cache protection exhausted: %d unprotected free blocks, allocation of %d fails",
                len(collected),
                num_blocks,
            )
            raise PinProtectedExhaustedError(f"only {len(collected)}/{num_blocks} unprotected free blocks available")
        skipped = [block for block, protected in popped if protected]
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
    logger.info("KV cache control enabled: hard eviction protection applied")


_apply_patch()
