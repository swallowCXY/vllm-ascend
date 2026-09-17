# KVCacheControlManager 设计（KV cache 生命周期控制）

> 分支基线：`feat/kv-cache-control-v0.25.1rc1`（vllm-ascend v0.25.1rc1 / vLLM 0.25.1）
> 本文描述**当前实现**的最终设计。历史演进（0.23.0 基线上的 P0/P1/P2 与需求收敛过程）见 git 提交记录与本文件旧版本；验收用例与边界细则见配套《验收规格书》（`KV_Cache_Control_Acceptance_Spec.md` v2.1）。

## 1. 背景与动机

Agent 场景对 KV cache 生命周期提出了当前 vLLM prefix caching（纯 LRU）无法表达的需求：

| 场景 | 诉求 | 现状问题 |
| --- | --- | --- |
| 多轮会话 / 长期系统提示词复用 | 内容 **pin 硬保护**，在 TTL 内绝不被 LRU 淘汰 | 与冷数据同等参与 LRU，热前缀易被大文档冲掉 |
| 会话级临时缓存 | KV 保留 **TTL**（如工具调用 10 分钟），到期自动解除保护 | 无过期概念，只能被动等 LRU 逐出 |
| sub agent 会话结束 | 收尾请求**主动释放**该请求的全部缓存注册 | 只能 `reset_prefix_cache` 全量清空，无法按内容释放 |
| 一次性大文档摘要 | 该请求内容**不写入缓存**（no store），避免污染缓存 | 无法按请求关闭 prefix cache 写入 |

### 1.1 设计目标

1. 新增 `KVCacheControlManager`（下称 KCCM），与上游 `vllm.v1.core.kv_cache_manager.KVCacheManager` **同级**，由 Scheduler 持有。
2. KCCM **只持有业务生命周期元数据**，**不持有 KV tensor**，**不直接修改/引用 `KVCacheBlock`、`BlockPool` 内部结构**；对块的影响全部经由 patch 与上游既有 API（`evict_blocks` 等）生效。
3. 三个面向 Agent 的控制能力：**pin（硬保护 + 必有 TTL）/ no-store / release**，统一在 message 上声明。
4. pin 池受配额约束，超限不激活并返回提示；明确 HBM 不足时的行为（§6）。

### 1.2 非目标（Non-goals）

- 不做物理回收与即时清零：release/TTL 到期只注销命中注册，数据在块复写时消失（"数据立即不可恢复"的合规语义需另行立项）。
- 不做外部 KV store 集成（tier/外部删除已移除）。
- 不做 namespace 级配额与 Prometheus 指标导出（后续项）。
- 不修改上游 vLLM 文件：全部干预通过 patch（`vllm_ascend/patch/`）实现。
- 不支持中段 token 范围的保存/删除（KV 因果性决定，见 §11.2）。

## 2. 功能定义与语义基线

三个能力统一在 message 上声明，**三 mode 互斥**（同请求出现 ≥2 种 mode 时全部不生效并返回提示）：

| mode | 声明位置 | 语义 |
| --- | --- | --- |
| `pin` | 恰好一个 message | 保护 **tools + messages[0..k]（含声明消息本身）**，尾部按块对齐截断；**硬性不淘汰**；TTL 必有（缺省 3600s，`ttl_s` 覆盖）；自请求**完成时**激活 |
| `no_store` | 任一 message 即整请求生效 | 本请求新产出的块不注册进前缀缓存；已命中的共享前缀不受影响 |
| `release` | 任一 message 即整请求生效 | 请求完成时**注销该请求块哈希覆盖的全部缓存注册**（含命中的共享未 pin 前缀） |

语义基线：

1. **内容寻址与块粒度**：缓存复用单位是块；所有边界按块向下取整，声明消息尾部的不完整块不保护。
2. **pin 硬保护 ≠ 无限保护**：受 TTL 约束；到期解除保护（非定时删除），回收由 LRU 决定。
3. **release 全删语义**：不做"仅删本请求新增"的区分；共享且未被对方 pin 的前缀会被连带注销（对方 miss 重算）。被覆盖的他人 pin 条目残留但失去保护对象；同内容被重新计算注册后，原 TTL 内会再次受保护。
4. **非物理回收**：块立即回到可分配队列，随时被新请求复用。
5. **互斥即全不生效**：混合 mode 的整条声明丢弃（含合法部分），以响应提示告知。
6. **易失性**：注册表/请求历史表为进程内存态，引擎重启失效。

## 3. 接口定义与调用方式

### 3.1 Message 级声明（唯一声明入口）

```jsonc
POST /v1/chat/completions
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<知识库/长文档>",
     "kv_cache_control": {"mode": "pin", "ttl_s": 7200}},   // ttl_s 可选，缺省 3600
    {"role": "user", "content": "<一次性大文档>",
     "kv_cache_control": {"mode": "no_store"}},
    {"role": "user", "content": "<sub agent 收尾>",
     "kv_cache_control": {"mode": "release"}}
  ]
}
```

| 字段 | 适用 mode | 说明 |
| --- | --- | --- |
| `mode` | 必填 | `pin` / `no_store` / `release` |
| `ttl_s` | pin 可选 | 保护时长秒数，缺省 3600 |

### 3.2 HTTP 控制面（仅 release）

```
POST /kv_cache/release
{"request_id": "chatcmpl-..."}          → {"released": true|false}
```

按已完成请求的 `request_id` 注销其全部缓存注册。依赖引擎内请求历史表（容量 `VLLM_ASCEND_KVCC_RELEASE_TABLE_SIZE`=4096，FIFO 淘汰；超出容量的旧请求返回 `released=false`）。无内置鉴权，生产由部署网关负责（与 `/reset_prefix_cache` 同信任级）。

### 3.3 配置项

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `VLLM_ASCEND_KV_CACHE_CONTROL` | `1` | 总开关；`0` 时所有声明被忽略，行为与基线一致 |
| `VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO` | `0.25` | pin 保护块数占总 GPU KV 块数上限比例；超限新 pin 不激活 + 日志提示 |
| `VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S` | `3600` | pin 默认 TTL |
| `VLLM_ASCEND_KVCC_RELEASE_TABLE_SIZE` | `4096` | HTTP release 可追溯的已完成请求数 |

### 3.4 观测通道

| 通道 | 用途 |
| --- | --- |
| 响应 `usage.prompt_tokens_details.cached_tokens`（需 `--enable-prompt-tokens-details`） | 命中判定主依据 |
| 非流式响应字段 `kv_cache_control_status` | 声明裁决结果：`{"status": "accepted"/"rejected", "reason": ...}`；rejected 原因：`conflict_modes` / `multiple_pin_messages` / `invalid_declaration` / `boundary_render_failed` |
| `GET /metrics` 的 `vllm:prefix_cache_hits/queries` | 交叉校验 |
| 服务端日志 | 配额超限（pin 超预算不激活）、TTL 过期等 |
| KCCM 内部计数 | `no_store_requests`/`pin_requests`/`release_requests`/`quota_degraded`/`ttl_expired`/`parse_errors`（暂未导出 Prometheus） |

## 4. 总体架构

```
+--------------------------------------------------------------------------+
| API server                                                                |
|   messages[].kv_cache_control ──> kv_cache_message.py（提取/裁决/边界计算） |
|         │ 注入 kv_transfer_params          │ 非流式响应附加 status          |
|         v                                 │                               |
|   OpenAIServingChat.render_chat_request（patch）                           |
+---------------------------┬----------------------------------------------+
                            v
| EngineCore / Scheduler                                                    |
|   KVCacheManager ── KCCM（元数据+策略）                                     |
|      │   ^  查询/计划(只读)                                                |
|      v   +---------------------------+                                    |
|   BlockPool <── patch: 硬保护驱逐过滤  │                                    |
|   KVCacheManager.allocate_slots/cache_blocks <── patch: no-store 跳过写入  |
|   KVCacheManager.free <── patch: pin 激活 / release 执行 / 历史记录         |
+--------------------------------------------------------------------------+
```

生效适配器（KCCM 产生决策，由现有所有权方执行，保证"不直接修改 block"）：

| 适配器 | 生效点 | 说明 |
| --- | --- | --- |
| A1 硬保护驱逐过滤 | `BlockPool.get_new_blocks` wrapper（`patch_kv_cache_eviction.py`） | 逐块弹出时查询 `protection_level`，保护块跳过并回队列；无法满足时恢复队列并抛 `PinProtectedExhaustedError` |
| A2 no-store 写入过滤 | `KVCacheManager.allocate_slots`（强制 `delay_cache_blocks=True`）+ `cache_blocks`（no-op）双 chokepoint | 覆盖全部调度器（Recompute/ProfilingChunk/Balance/上游基类），无需 scheduler 钩子 |
| 生命周期钩子 | `KVCacheManager.free` wrapper | 请求完成：pin 激活（配额判定）/ release 计划执行（`evict_hashes`）/ 请求历史记录 |

## 5. 实现设计

### 5.1 KCCM 元数据模型（`core/kv_cache_control_manager.py`）

```python
@dataclass
class PinEntry:
    request_id: str        # 注册键
    hashes: frozenset      # 保护的块哈希集合
    expire_at: float       # monotonic；TTL 必有

class KVCacheControlManager:
    _pin_entries: dict[request_id, PinEntry]       # 注册表
    _hash_index: dict[hash, set[request_id]]       # 反查索引，protection_level O(1)
    _request_history: dict[request_id, hashes]     # 已完成请求表（FIFO 上限），支撑 HTTP release
    _release_plan: list[hashes]                    # 待执行注销计划（由 free wrapper 消费）
```

API：`parse_request_control`（解析+memoize 到 Request 属性）、`is_no_store`/`is_release`、`on_request_finished`、`release_request`、`take_release_plan`、`protection_level`（0/1）、`has_protection`（快路径）、`maybe_sweep`（惰性 TTL 过期，挂 `allocate_slots` wrapper，`next_expiry` O(1) 快路径）、`bind_kv_cache_manager`（绑定配额分母与 block_size）。

声明解析：serving 层聚合后引擎只见单 mode + 可选 `pin_boundary_tokens`/`ttl_s`；pin 的 TTL 缺省取 `VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S`。

### 5.2 生效适配器

**A1 硬保护驱逐过滤**（`patch_kv_cache_eviction.py`）：wrap `BlockPool.get_new_blocks`。无保护时走原路径；有保护时逐块弹出，`protection_level > 0` 的块暂存；集齐后暂存块回队列尾部（LRU 刷新）。若队列耗尽仍未集齐：**按弹出原序恢复整个队列**并抛 `PinProtectedExhaustedError`——保护块绝不参与驱逐。

**A2 no-store 双 chokepoint**（`patch_kv_cache_control.py`）：`allocate_slots` 对 no-store 请求强制 `delay_cache_blocks=True`（复用上游 P/D 异步加载语义），`cache_blocks` 直接 no-op。位置参数守卫处理 `delay_cache_blocks` 的第 6 位传参。共享前缀已注册的哈希不受影响（对标 Claude Code `skipCacheWrite`：只读不写）。

**生命周期钩子**（`KVCacheManager.free` wrapper）：请求完成时依次执行 KCCM `on_request_finished`（pin 激活含配额硬判定 / release 出队注销计划 / 历史记录）→ `take_release_plan` → `evict_hashes`（跨 group 哈希→块解析，复用上游 `BlockPool.evict_blocks`，自动发 `BlockRemoved` KV 事件）。`allocate_slots` wrapper 捕获 `PinProtectedExhaustedError` 并返回 `None` → 走上游正常 preemption 路径。

### 5.3 serving 层（`entrypoints/kv_cache_message.py`）

- **提取**：兼容 dict 与 pydantic（`model_extra`/getattr）消息形态；非法 mode 记 `invalid_declaration`。
- **裁决**：≥2 种 mode → 全部不生效（`conflict_modes`）；pin message 多于一个 → 不生效（`multiple_pin_messages`）。
- **边界计算**：对 pin message k，增量渲染 `messages[:k+1]`（含 tools 与模板参数）→ 前缀单调校验（prefix 渲染串必须是 full 渲染串的字符串前缀，否则 `boundary_render_failed`）→ boundary_tokens。
- **注入**：写入 `request.kv_transfer_params["kv_cache_control"]`（挂点为 `OpenAIServingChat.render_chat_request` wrapper，早于 `to_sampling_params` 构建，API 层零改动）。
- **提示**：status 暂存于 request，`create_chat_completion` wrapper 附加到非流式响应 `kv_cache_control_status` 字段（流式仅日志）。

### 5.4 控制面（仅 release）

`EngineCore.kv_cache_release(request_id)` 经既有 `call_utility` 反射通道（`EngineCoreRequestType.UTILITY` → `getattr(self, method_name)`）到达；`AsyncLLM.kv_cache_control_async` 为通用入口；`build_app` wrapper 挂载 `/kv_cache/release` 路由。不含任何会话后修改 pin/TTL 的能力（需求决策）。

### 5.5 块状态机与 release 语义

```
运行中 (ref_cnt>0，不在 free 队列，不可分配)
   │ 请求结束 free()                    ← 此刻块即回队列，可随时重分配
   ▼
空闲+可命中 (在 free 队列，哈希已注册)  ← 命中触发 touch 刷新位置
   │ release() = 注销哈希（变匿名，仍在队列）
   ▼
空闲+匿名 (在 free 队列，未注册)       ← 随时可分配，驱逐首选
   └──────────────→ 被新请求取出重新分配
```

- "释放后可重分配"在请求 `free()` 时刻即成立；release 只提前终止可命中状态。
- 保护与可分配性正交：pin 保护只影响受害者选择，从不把块移出队列。
- release 非物理回收：数据在块复写时消失。

## 6. HBM 不足时的策略

1. **硬保护**：pin 块在 TTL 内绝不参与驱逐（无软兜底）。
2. **配额准入**：pin 池上限 = `PIN_BUDGET_RATIO` × 总 GPU KV 块数；激活时判定（`len(hash_index) + 新增 ≤ 预算`），超限不激活 + warning + `quota_degraded` 计数。
3. **保护耗尽时的行为**：free 队列中非保护块不足以满足分配时，`PinProtectedExhaustedError` → `allocate_slots` 返回 `None` → 上游抢占路径。后果：pin 池用满配额后，极端压力下非 pin 流量可用容量压缩至配额外部分，**可能触发正常请求抢占**（已确认接受；不会死锁——被抢占块非 pin，回到可分配队列，系统收敛于"pin 池 + 其余容量服务正常流量"）。
4. **no-store 天然不占 HBM**；TTL 过期条目经惰性 sweep 解除保护（`ttl_expired` 计数）。

## 7. 功能边界（明确不保证的行为）

1. pin 硬保护 ≠ 无限保护：受 TTL（缺省 1h）与请求完成时激活约束。
2. release 全删：共享未 pin 前缀连带注销（需求方需认可，见验收规格书 AC-REL-04）。
3. 非物理回收/无即时清零。
4. 块粒度 + 边界最佳努力（模板非前缀单调 → 声明不生效并提示）。
5. 互斥即全不生效；配额超限仅日志提示（无同步响应提示）。
6. 流式响应无 status 字段（仅日志）。
7. 控制面仅 release；记录表容量限制；无内置鉴权。
8. 无 external store 集成；无 Prometheus 导出；重启易失。
9. 模型范围：标准 prefix caching 模型（FullAttention 族）；hybrid/SWA 未验证。
10. 上游锚点依赖 vLLM 0.25.1（见 §11.3），升级需回归。

完整边界细则与验收判据见《验收规格书》§5。

## 8. 风险与开放问题

| # | 风险 | 应对 |
| --- | --- | --- |
| 1 | `get_new_blocks`/`allocate_slots` patch 与上游漂移 | patch 面最小化 + marker 幂等 + feature flag；升级时锚点复核（§11.3 清单） |
| 2 | release 全删误伤共享内容 | 已确认为需求语义；被覆盖的他人 pin 条目在原 TTL 内对重算内容继续生效；文档与验收用例（AC-REL-04）显式标注 |
| 3 | 硬保护引发非 pin 流量抢占 | 已确认接受；`quota_degraded`/日志可观测；配额比例可调 |
| 4 | 消息边界计算依赖模板单调性 | 渲染串前缀校验，不满足即拒绝并提示；不影响请求本身 |
| 5 | TTL 精度 | monotonic 时钟 + 惰性清扫，秒级误差（一个调度步内） |
| 6 | 元数据/历史表内存膨胀 | pin 池受配额约束；历史表 FIFO 上限（4096） |
| 7 | DP/多副本 | release 控制命令与 `reset_prefix_cache` 同链路扇出，多 DP 行为待容器专项确认 |
| 8 | 权限与滥用 | `/kv_cache/release` 无内置鉴权，依赖部署网关；无 namespace 配额（后续项） |
| 9 | hybrid/SWA/压缩模型 | 未验证；`protection_level` 对未覆盖场景保守返回（无保护） |
| 10 | 重启丢失 | 注册表/历史表内存态；重启后需重新声明 |
| 11 | 热路径性能 | 钩子 O(1)；sweep 摊还；UT 与容器回归覆盖（AC-PERF-01/02） |
| 12 | P/D disaggregated 场景 | `delay_cache_blocks` 原有语义与 no-store 组合需容器专项验证一次 |

## 9. 测试与验收

- **UT（83 个，本地全绿）**：核心 20（解析矩阵/pin 生命周期/release/history 表/sweep）、wrapper 12（no-store 双 chokepoint/free 钩子/幂等/env 开关）、驱逐过滤 6（保护跳过/耗尽抛错/队列恢复）、控制面 8（release 反射/路由/服务 patch）、消息级 18（提取/裁决/边界/单调校验/注入/提示）、ascend_store 9（no-store 跳过点）。
- **回归**：ascend_store 全套 186 passed（v0.25.1rc1 基线）。
- **容器验收**：按《验收规格书》§4 执行（v0.23.0 基线的 no-store E2E 实测记录已存档于本文件 git 历史）；待办：vLLM 0.25.1 镜像下三链路专项回归。

## 10. 分期与历史

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 请求级 no-store（双 chokepoint） | 0.23.0 基线实现并容器 E2E 通过（历史存档） |
| P1/P2 | 软 pin + 引用计数 + 控制面 + 外部存储集成 | 被 v2 需求收敛**取代** |
| v2 | message 级三 mode（pin 硬保护+TTL 合一 / release 全删 / 互斥裁决） | 当前实现 |
| 迁移 | 0.23.0 → 0.25.1rc1（cherry-pick + 锚点预验证） | 当前分支 |

## 11. 附录

### 11.1 业界参照

**Claude Code（客户端侧）**：断点式 `cache_control`（message/content block 上）与消息级声明形态同源；`skipCacheWrite`（断点移到倒数第二条消息 = 只写共享前缀）与 no-store 语义等价；`tengu_cache_eviction_hint`（conversation_clear/session_end/subagent_end）只能提示、由服务端执行——本设计的 release 即其服务端执行者；TTL 由服务商标管理，本设计引擎自主。会话稳定锁存（TTL 资格/tool schema/beta header 会话级锁定）是贯穿性纪律：**会话中途不翻转缓存键**。

**上游 vLLM**：驱逐纯 LRU、无优先级/保护位/TTL 概念（`BlockPool`），唯一保护是 `ref_cnt>0`；`skip_reading_prefix_cache` 只跳读、无跳写——no-store 补的是上游空白。`evict_blocks` + `BlockRemoved` 事件、`call_utility` 反射通道、`cache_salt` 第 0 块隔离机制均为本设计直接复用的既有设施。Claude Code 的 cached microcompact（`cache_edits` 中段删除）在 vLLM 内容寻址模型下无对应机制，以 release+重新预热近似。

### 11.2 token 级与块对齐约束

KV 因果性（token i 的 KV 依赖全部前序 token）决定了只有**前缀**可复用，缓存物理单位是块：

- token 级控制只能表达为"块对齐的前缀长度"（消息级声明经边界计算后同样 floor 到块）。
- 非块对齐案例（137 token system prompt，bs=128）：0~128 可保护，尾部 9 token 每请求重复 prefill（成本可忽略）；全覆盖需内容补齐到块边界或减小 block_size。
- 中段删除物理不可行（删 B 后 C 的 KV 全部失效）；Anthropic 服务端的 `cache_edits` 中段删除在 vLLM 无对应，以 release+重预热近似。
- 共享段结束位置必须落在块边界内——把含用户内容的块一并 pin 不等于该内容级的共享保护。

### 11.3 上游锚点验证清单（v0.25.1，迁移时逐一确认存活）

| 锚点 | v0.25.1 位置 |
| --- | --- |
| `KVCacheManager.allocate_slots(delay_cache_blocks)`（位置参数序不变） | kv_cache_manager.py:248 |
| `KVCacheManager.cache_blocks` / `free` / `evict_blocks` | :620 / :466 / :508 |
| `KVCacheManager.block_pool` / `kv_cache_config` | :161 |
| `BlockPool.get_new_blocks` / `evict_blocks` / `get_num_free_blocks` / `free_block_queue` / `cached_block_hash_to_block.get_one_block` | block_pool.py:542/637/692 |
| `make_block_hash_with_group_id` / `get_block_hash` | kv_cache_utils.py:57/69 |
| `Request.block_hashes` / `kv_transfer_params` | request.py:179/115 |
| UTILITY 反射 `getattr(self, method_name)` / `call_utility_async` | core.py:1393 / core_client.py:1101 |
| `AsyncLLM.engine_core` | async_llm.py |
| `api_server.build_app` | api_server.py:157 |
| `OpenAIServingChat.render_chat_request` → `to_sampling_params` 时序（render 后、采样构建前） | serving.py:206→318 |
| `ChatCompletionRequest.kv_transfer_params` 顶层字段 / `OpenAIBaseModel` extra="allow" | protocol.py |
