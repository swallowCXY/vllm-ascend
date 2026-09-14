# KVCacheControlManager 初步设计

## 1. 背景与动机

Agent 场景对 KV cache 生命周期提出了当前 vLLM prefix caching（纯 LRU）无法表达的需求：

| 场景 | 诉求 | 现状问题 |
| --- | --- | --- |
| 多轮会话 / 长期系统提示词复用 | 内容**软 pin**，长期保存，优先不被 LRU 淘汰 | 与冷数据同等参与 LRU，热前缀易被大文档冲掉 |
| 会话级临时缓存 | KV 保留 **TTL**（如 30 分钟），到期自动失效 | 无过期概念，只能被动等 LRU 逐出 |
| 会话结束 | Agent 主动调用**提前释放**指定 KV | 只能 `reset_prefix_cache` 全量清空，无法按内容释放 |
| 一次性大文档摘要 | 该请求内容**不写入缓存**（no store），避免污染缓存 | 无法按请求关闭 prefix cache 写入 |

### 1.1 设计目标

1. 新增 `KVCacheControlManager`（下称 KCCM），与上游 `vllm.v1.core.kv_cache_manager.KVCacheManager` **同级**，由 Scheduler 持有。
2. KCCM **只持有业务生命周期元数据**（内容标识 → 保留策略），**不持有 KV tensor**，**不直接修改/引用 `KVCacheBlock`、`BlockPool` 内部结构**。
3. 提供四个面向 Agent 的控制接口：soft pin / TTL / 主动释放 / no-store。
4. 对 HBM 不足场景给出明确的驱逐与准入策略（软 pin 语义，见 §7）。

### 1.2 非目标（Non-goals）

- 不做硬 pin（不保证任何情况下内容一定驻留 HBM）。
- 不实现 KV tensor 的搬运/压缩（demote 到外部存储复用现有 AscendStore/KVPool 连接器）。
- 不修改上游 vLLM 文件，所有对上游行为的干预通过 patch（`vllm_ascend/patch/`）或组合实现，符合插件架构规范。

## 2. 总体架构

```
+---------------------------------------------------------------+
| Scheduler (upstream Scheduler / 本仓库 RecomputeScheduler 等)  |
|                                                                |
|  self.kv_cache_manager          self.kv_cache_control_manager |
|  (KVCacheManager, 上游)          (KCCM, 新增, 元数据+策略)      |
|         |    ^  查询/计划(只读)          ^    |                 |
|         |    +--------------------------+    | 生命周期事件      |
|         v                                    v                 |
|   BlockPool(块+LRU)  <--- patch: 驱逐时咨询 KCCM                |
|   KVCacheManager.cache_blocks <--- patch: no-store 跳过注册     |
|         |                                                      |
|         v                                                      |
|   KVConnector (AscendStore/KVPoolScheduler ...)                |
|   <--- connector metadata 扩展: TTL/pin/no-store 透传外部存储    |
+---------------------------------------------------------------+
        ^                                    ^
        |  请求级声明 (kv_transfer_params, 随请求流动)   |  控制面 RPC (pin/release/set_ttl)
   Agent / API server <------------------------+
```

三个**生效适配器**（KCCM 产生决策，由现有所有权方执行，保证"不直接修改 block"）：

| 适配器 | 生效点 | 说明 |
| --- | --- | --- |
| A1 驱逐过滤 | `BlockPool` 驱逐路径 patch（`patch/platform/patch_kv_cache_eviction.py`，新增） | 驱逐候选选择时调用 `KCCM.protection_level(block_hash)` 跳过受保护块 |
| A2 写入过滤 | `KVCacheManager.cache_blocks` wrapper patch | no-store 请求的新满块**不注册**进 `cached_block_hash_to_block`，直接释放为普通空闲块 |
| A3 连接器透传 | `AscendConnectorMetadata` / `ReqMeta` 扩展 | 对外部 KV Pool：put 时携带 TTL/pin/no-store 标记；release 触发异步 delete；TTL 可用后端原生 TTL（如 Mooncake） |

KCCM 不持有对 `BlockPool` / tensor 的引用，只暴露**只读查询**与**决策输出**，由 patch 侧拉取。

## 3. 内容标识与元数据模型

### 3.1 内容标识（ContentRef）

vLLM prefix cache 以 `BlockHash` 链（内容哈希）寻址。Agent 视角的内容引用支持三种形式：

```python
@dataclass(frozen=True)
class ContentRef:
    namespace: str = "default"          # 隔离域，用于配额与权限
    request_id: str | None = None       # 引用某请求产出的全部前缀
    prefix_len: int | None = None       # 可选：只引用前 prefix_len 个 token（block 对齐）
    cache_key: str | None = None        # Agent 自定义业务键（与 cache_salt 拼接后参与哈希）
```

- `request_id` 引用：请求结束时由 scheduler 钩子捕获该请求的 `BlockHashList` 前缀，展开为哈希集合存入元数据（哈希是元数据，不违反"不持有 tensor"）。
- `cache_key` 引用：要求请求带同名 `cache_key`（等价于上游 `cache_salt` 机制扩展），命中时按 key 反查。
- 归一化：任意引用最终解析为 `(namespace, root_block_hash, num_blocks)`，便于 ref-count 与共享前缀去重。

### 3.2 生命周期条目与状态机

```python
class RetentionPolicy(Enum):
    NO_STORE = "no_store"    # 请求级：内容不写缓存
    NORMAL = "normal"        # 默认 LRU
    TTL = "ttl"              # 到期回落 NORMAL
    PINNED = "pinned"        # 软 pin，可带到期时间

@dataclass
class LifecycleEntry:
    ref: tuple[str, BlockHash, int]   # (namespace, root_hash, num_blocks)
    block_hashes: frozenset[BlockHash]  # 展开后的哈希集合（元数据）
    policy: RetentionPolicy
    priority: int                     # pin 优先级，越大越晚被逐
    pin_count: int                    # 多个 Agent pin 同一内容时的引用计数
    expire_at: float | None           # monotonic 时钟
    created_at: float
    request_ids: set[str]             # 溯源
```

状态迁移：

```
 请求完成(finish hook)                    sweep / 主动 set_ttl
NORMAL ─────────────► PINNED/TTL ◄──────────────► (延长/缩短)
  ▲                        │ expire_at 到期
  │                        ▼ 回落 NORMAL（哈希集合随后按 LRU 自然淘汰）
  └── release() 主动释放 ──► 注销元数据 + 触发底层删除(A2 释放注册/A3 外部 delete)
```

### 3.3 存储结构

- `dict[tuple[BlockHash], LifecycleEntry]`（前缀锚点 → 条目）+ `dict[BlockHash, WeakSet[entry]]`（块哈希 → 所属条目，供 `protection_level` O(1) 查询）。
- `dict[str, LifecycleEntry]`（request_id → 条目，no-store 与 TTL 注册用）。
- 元数据内存上限受 pin/TTL 配额约束（§7.3），防止哈希集合膨胀。

## 4. 四个 Agent 接口设计

### 4.1 接口签名（KCCM 内部 API，即 EngineCore 侧实现）

```python
class KVCacheControlManager:
    # ---- Agent 控制面（经 EngineCore RPC 暴露，见 4.5）----
    def pin(self, ref: ContentRef, *, priority: int = 0,
            ttl_s: float | None = None) -> PinHandle:
        """软 pin 指定内容。返回句柄用于后续 release。
        - ttl_s 同时给定时为 pin 带过期时间
        - 内容尚不存在时登记为 pending，请求完成后自动生效
        - 超出 pin 配额时：报 PinQuotaExceededError（由 API 层映射 409 或降级为 TTL）"""

    def set_ttl(self, ref: ContentRef | PinHandle, ttl_s: float | None) -> None:
        """设置/调整/取消(None) TTL。已 PINNED 条目设置 TTL 表示到期自动解除 pin。"""

    def release(self, ref: ContentRef | PinHandle) -> bool:
        """提前释放：pin_count-1；归零则注销元数据并触发删除计划。
        返回 False 表示引用不存在（幂等，不报错）。"""

    def set_no_store(self, request_id: str) -> None:
        """请求级兜底入口：标记该请求产出内容不写缓存。
        首选路径是随请求声明（4.2），此接口用于请求已发出后的补救/内部调用。"""
```

### 4.2 Agent 实际调用形态

**请求级声明（首选，随请求原子生效，无竞态）**——通过 OpenAI 兼容层请求体的顶层字段 `kv_transfer_params`（OpenAI SDK 客户端可用 `extra_body` 等价透传，SDK 会将其合并进顶层 body）：

```jsonc
POST /v1/chat/completions
{
  "messages": [...],
  "kv_transfer_params": {
    "kv_cache_control": {
      "mode": "no_store"                      // 本请求内容不入缓存
      // "mode": "pin",  "priority": 10, "ttl_s": 86400
      // "mode": "ttl",  "ttl_s": 1800
      // "cache_key": "session-abc-42"        // 可选业务键，便于后续 pin/release
    }
  }
}
```

**会话后控制（pin 已有内容 / 延长 TTL / 主动释放）**——API server 新增路由，转发为 EngineCore 控制命令（沿用 `reset_prefix_cache` 这类引擎级控制操作的既有通道）：

```http
POST /kv_cache/pin      {"cache_key": "session-abc-42", "priority": 10, "ttl_s": 86400}
POST /kv_cache/ttl      {"cache_key": "session-abc-42", "ttl_s": 3600}
POST /kv_cache/release  {"cache_key": "session-abc-42"}     # agent 会话结束钩子调用
```

### 4.3 四接口语义要点

| 接口 | 关键语义 | 错误/边界 |
| --- | --- | --- |
| `pin` | 软 pin：仅影响**无引用块的驱逐顺序**，绝不阻塞活跃请求分配；ref-count 支持多方叠加；优先级支持分层 | 超配额 → 拒绝或降级 TTL；内容不存在 → pending 直到产生 |
| `set_ttl` | TTL 是**到期回落 NORMAL**而非硬删除（HBM 侧由 LRU 决定何时真正回收；外部存储侧可透传硬 TTL） | 已过期/不存在的 ref → 幂等成功 |
| `release` | 主动提前释放；释放的是"缓存资格"，正在被活跃请求引用的块不受影响（ref_cnt 语义不变，KCCM 不碰块） | 幂等；跨 namespace 拒绝（权限） |
| `set_no_store` | 只约束**本请求新增满块**；已存在的共享前缀块不受影响（内容寻址决定了无法"反存"共享块） | 必须在请求 cache_blocks 之前生效，见 4.4 竞态讨论 |

### 4.4 Scheduler 侧生命周期钩子（KCCM 内部接口，热路径）

```python
# 以下均为 O(1) 或摊还 O(1)，禁止在热路径做集合展开等重操作
def on_request_scheduled(self, request) -> None:
    """读请求级 kv_cache_control 声明；no-store → 标记，pin/ttl → 登记 pending"""

def on_request_finished(self, request, block_hashes: BlockHashList) -> None:
    """请求结束：no-store → 清标记；pin/ttl → 用捕获的哈希链展开元数据并生效"""

def is_no_store(self, request_id: str) -> bool: ...          # A2 cache_blocks wrapper 调用
def protection_level(self, block_hash: BlockHash) -> int: ... # A1 驱逐过滤调用，返回 0=无保护
def sweep_expired(self) -> None: ...                          # 每个调度步开头惰性清扫（降级+删除计划）
def take_release_plan(self) -> ReleasePlan | None: ...        # 供 A2/A3 消费（注销注册/外部 delete）
```

挂点位置（本仓库现状）：

| 钩子 | 挂点 |
| --- | --- |
| 构造 | patch `Scheduler.__init__`（feature flag 控制），与本仓库各 Scheduler 子类兼容 |
| `on_request_scheduled` | `schedule()` 入口 patch（覆盖 ProfilingChunk/Recompute/Balance/DynamicBatch 四个实现） |
| `on_request_finished` | `finish_requests` / `free` 路径 patch |
| no-store 生效 | `KVCacheManager.cache_blocks` wrapper（调用点如 `recompute_scheduler.py:150`、`scheduler_profiling_chunk.py`） |
| 驱逐过滤 | `BlockPool` 驱逐路径 patch（唯一需要理解块回收语义的 patch） |
| 外部存储 | `KVPoolScheduler`（`pool_scheduler.py`）的 metadata 组装处透传标记 |

## 5. 与外部 KV Pool 的联动（A3）

本仓库 AscendStore / Mooncake / LMCache / UCM 连接器已存在：

- **put 时**：`ReqMeta` 增加 `lifecycle: {policy, ttl_s, priority}`；后端支持原生 TTL 的（如 Mooncake `put(..., ttl)`）直接透传，不支持的由 KCCM sweep 产生 delete 计划补偿。
- **release 时**：`ReleasePlan` 经调度步下发 → 连接器异步 delete 外部键。
- **no-store 时**：put mask 置空（`AscendStoreCoordinator.store_mask` 侧已有 mask 机制，透传跳过即可）。
- 与现有 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`（`pool_scheduler.py:94`）的关系：retention_interval 是全局静态策略，KCCM 是按内容粒度的动态策略，两者叠加取更严格者。

## 6. DP / 多实例

- KCCM 实例随每个 EngineCore（每个 DP rank）各一份，控制命令由 API server 层扇出到全部 rank，要求**幂等**（同一命令重复送达结果不变）。
- pin/TTTL 元数据为进程内状态，rank 间不共享；Agent 需路由到正确实例时依赖请求级声明优先。
- release 扇出部分失败：按幂等重试收敛，不做分布式事务。

## 7. HBM 不足时的策略（核心风险应答）

### 7.1 基本原则

1. **活跃请求永远优先**：pin 保护只作用于"已 `free`、引用计数为 0、可被驱逐复用"的缓存块，绝不阻塞 `allocate_slots`。KCCM 引入的任何保护都不允许导致 preemption。
2. **软 pin 语义**：`PINNED` 提升的是驱逐优先级（越晚被逐），不是绝对驻留承诺。极端情况下高优先级 pin 块也可被逐（逐出前优先 demote 到外部存储，若配置了连接器）。

### 7.2 驱逐顺序（A1 patch 的候选选择规则）

```
NORMAL(LRU 最久未用)  →  TTL 已过期  →  PINNED 低优先级  →  PINNED 高优先级
                                                   （每一级内部仍按 LRU）
```

no-store 内容从不进入缓存，天然不占 HBM。驱逐受保护块时若配置外部存储，先写后逐（demote），TTL 透传。

### 7.3 准入控制（防止 pin 满仓）

- **总量配额**：pin 保护池上限 = HBM KV 总量的固定比例（env：`VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO`，建议默认 0.25，写入 `envs.py` 走环境变量评审流程）。超配额 → 新 pin 请求返回 `PinQuotaExceededError`，API 层可降级为长 TTL。
- **namespace 配额**：单 Agent/租户上限，防单点占满。
- **高水位主动清扫**：被 pin 保护的空闲块占比超水位时，sweep 优先降级最老/最低优先级 TTL 条目。
- **不变量**：任意时刻受保护空闲块数 ≤ 配额块数，从而保证正常负载下驱逐候选永远非空，`allocate_slots` 失败率不因 pin 恶化。

### 7.4 降级矩阵

| 条件 | 行为 |
| --- | --- |
| HBM 充足 | pin 内容常驻，TTL 到期回落 LRU |
| HBM 紧张（未超配额） | 受保护块被跳过，逐 NORMAL/过期内容 |
| 超过 pin 配额 | 新 pin 请求拒绝（409）或降级为 TTL |
| 有外部 KV Pool | 受逐 pin 块 demote 到外部存储而非丢弃；release 同步删外部键 |
| 无外部 KV Pool | demote 退化为直接丢弃，pin 仅在 HBM 内部生效 |

## 8. 风险与开放问题

| # | 风险 | 应对 |
| --- | --- | --- |
| 1 | **BlockPool patch 与上游漂移**：驱逐路径 patch 是全方案中最脆的耦合点 | patch 面积最小化（只改候选选择处），加版本兼容 try/except，feature flag 默认关闭；评审按 AGENTS.md 严格架构评审 |
| 2 | **共享前缀所有权**：多个请求共享同一 block 哈希链，一方 no-store/release 不能影响他方 | 内容寻址天然去重：release 只减 pin_count；no-store 只跳过"本请求新满块"注册，已在缓存中的共享块不动；文档中明确此语义 |
| 3 | **no-store 与 chunked prefill 竞态**：`set_no_store` 到请求第一次 `cache_blocks` 之间有窗口 | 请求级声明随请求对象原子到达，无窗口；控制面补救接口仅尽力而为，文档标注 |
| 4 | **release 与外部 load 在途竞态**：正从外部存储 load 的内容被 release | release 只标记删除计划，连接器侧按代际（generation id）忽略过期结果 |
| 5 | **TTL 时钟与精度** | 一律 monotonic 时钟；惰性清扫精度为一个调度步，接受秒级误差；外部存储用后端原生 TTL 兜底 |
| 6 | **元数据膨胀**：pinned 前缀哈希集合可很大 | 配额约束（§7.3）；哈希集合惰性展开 + memoize；元数据内存计入监控指标 |
| 7 | **DP/多副本路由**：pin 打到非目标 rank 缓存命中率失效 | 请求级声明优先；控制面扇出全 rank；会话亲和由上层网关负责（开放问题） |
| 8 | **权限与滥用**：Agent A 释放 Agent B 的 pin；恶意 pin 占满配额 | namespace 隔离 + API server 鉴权 + namespace 配额 |
| 9 | **hybrid/SWA/压缩模型兼容性**：SWA 块哈希链不完整、压缩族键粒度不同，pin 语义可能失效 | 首期只对 FullAttention 家族启用；`protection_level` 对其余 family 恒返回 0（coordinator 的 cache_family 机制可判断） |
| 10 | **重启丢失**：pin 注册表在内存 | 明确为易失语义；后续可选：持久化 pin 清单 + 启动时按 key 重 pin（依赖外部存储） |
| 11 | **热路径性能**：每请求/每步新增 dict 操作 | 钩子全部 O(1)；sweep 摊还；benchmark 前缀命中场景无回归；纯 CPU 侧，无 `tensor.item()`/同步问题 |
| 12 | **prefix 命中统计口径**：pin 造成的"非 LRU 存活"影响命中率指标解读 | 新增指标：`pinned_blocks`、`pin_evictions`、`ttl_expired`、`no_store_requests`、`release_calls` |

**开放问题**：

1. `cache_key` 是否复用/扩展上游 `cache_salt` 机制，还是独立一层映射？倾向独立映射，避免侵入哈希计算。
2. release 后立即释放还是标记"待回收"（给在途请求一个宽限步）？初步倾向标记待回收，由 sweep 处理。
3. 与 Disaggregated Prefill（本仓库已有设计文档）下 P/D 两侧生命周期不对称如何对齐，需后续专项设计。

## 9. 分期实施

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| P0 | KCCM 元数据类 + 请求级 no_store（A2 patch）+ TTL 登记/惰性 sweep + 指标 | `vllm_ascend/core/kv_cache_control_manager.py`、patch、UT |
| P1 | soft pin + A1 驱逐过滤 patch + 配额/水位 | patch、UT（驱逐策略矩阵） |
| P2 | 控制面 RPC + API server 路由 + release + A3 连接器透传（外部 TTL/demote/delete） | E2E |

## 10. 测试计划

- **UT**（`tests/ut/core/`）：元数据状态机；四接口语义与幂等；驱逐顺序矩阵；配额拒绝/降级；共享前缀释放 ref-count；TTL 惰性清扫。
- **UT（patch 兼容性）**：`tests/ut/` 中 mock 上游 `BlockPool`（参照 `tests/ut/distributed/ascend_store/_mock_deps.py` 模式）验证 A1/A2 行为。
- **E2E**（`tests/e2e/`）：pin 内容在持续压力下命中；no-store 不留缓存痕迹（prefix stats 验证）；release 后同前缀二次请求 miss；外部存储 TTL/删除联动。
- **性能**：调度热路径回归（`prefix_cache_stats` 与吞吐对比），确认钩子开销可忽略。

## 11. 附录：设计问答

### 11.1 no-store 与 chunked prefill 竞态具体指什么

chunked prefill 将长请求拆成多个调度步，每步的 `allocate_slots`/`cache_blocks` 都会把"新写满的块"注册进前缀缓存。竞态只发生在**控制面补救路径**（`set_no_store(request_id)` 为异步 RPC，agent 在流式响应开始后才能拿到 request_id）：

```
调度步1(注册块0..N) → 调度步2(注册块N..M) → ... → 控制命令到达
                                            ↑ 头部块已注册，no-store 只能部分生效
```

请求级声明（顶层 `kv_transfer_params` 随 Request 对象）在首次 `schedule()` 前即被 `on_request_scheduled` 读取，无窗口。缓解手段：请求级优先；控制面标注 best-effort；可选硬化——KCCM 记录该请求已注册的哈希集合，晚到命令触发"追补注销"（跳过共享块）。

### 11.2 Agent 与推理接口的交互通道

1. **请求内通道（首选，原子）**：HTTP 顶层字段 `kv_transfer_params` → API server 解析 → `Request` 对象字段 → EngineCore → scheduler `on_request_scheduled` 读取。适用于随请求声明 pin/ttl/no-store。
2. **会话外通道（控制面）**：HTTP 路由 → `AsyncLLM` → EngineCore RPC（沿用 `reset_prefix_cache` 等引擎级控制操作的既有通道）→ KCCM；DP 场景扇出全 rank，幂等收敛。适用于对**已产生内容**的 pin/set_ttl/release。

### 11.3 "token 级 / 内容级"控制的设计方案

**根本约束**：KV 具有因果性（token i 的 KV 依赖全部前序 token），因此只有**前缀**可复用；缓存物理单位是块。token 级控制只能下取整为"块对齐的前缀长度"；中段删除无意义（删除 B 后 C 的 KV 全部失效）。"token 级"实际表达为"前缀长度级 + 内容键级"。

**L1 引擎内门面（推荐）**——agent 感知不到 block：

```python
ContentRef(namespace, cache_key, prefix_tokens)   # prefix_tokens 由 KCCM 向下取整到块对齐
pin(ref, tier="hbm" | "external", ttl_s, priority)  # tier 决定驻留层级
release(ref)                                        # 按业务 key 删除，HBM 注销 + 外部 delete
```

`tier="external"` 时长期内容落外部 KV Pool（AscendStore/Mooncake 原生 TTL 与按 key delete），HBM 仅作加速层——**长期保存与精确删除的正确归宿是外部层**，与 HBM 配额解耦。

**L2 曲线等效（不改引擎，agent 框架侧）**：

| Agent 意图 | 曲线方案 | 局限 |
| --- | --- | --- |
| pin 长期内容 | 前缀组装：长期内容固定放请求头，预热时发 prefill-only 请求（`max_tokens=1`）写入缓存 | 需 agent 编排配合 |
| pin/TTL | LRU 心跳保活：定时发送 mini 请求刷新前缀的 LRU 顺序 | 有周期性计算开销，非真正 TTL |
| 内容隔离 | `cache_salt` 命名空间隔离 | 无删除能力 |
| 长期保存/精确删除 | 外部 KV Pool 原生 TTL / 按 key delete | 需连接器 |
| no-store | **无干净曲线方案**：最接近的是独立 `cache_salt` + 接受 LRU 自然淘汰 | 污染该命名空间 |

**L3 "精确 token 段"控制**：因 KV 因果性物理不可行；最接近的替代是把想分别管理的内容段拆成多个独立请求，各自作为前缀缓存，推理时按序拼接（依赖前缀拼接处的哈希连续性）。

### 11.4 Agent 意图 → 机制映射速查

| Agent 想做的事 | 机制 |
| --- | --- |
| 系统提示词/知识库长期复用 | 请求级 `mode=pin` 或 warmup + `pin(cache_key=...)` |
| 会话缓存限时保留 | 请求级 `mode=ttl, ttl_s=N` 或 `set_ttl` |
| 会话结束清理 | 会话钩子调 `POST /kv_cache/release` |
| 一次性大文档不污染缓存 | 请求级 `mode=no_store` |
| 删除某份指定文档的缓存 | `pin` 时绑定 `cache_key`，之后 `release(cache_key)` |
| 跨重启长期保存 | `tier="external"`，落外部 KV Pool，删除走存储原生接口 |

### 11.5 案例：非块对齐的 system prompt（137 token，block_size=128）

```
块0: token 0..127    ← 满块，哈希只依赖 0..127 → 可 pin，跨请求共享 ✅
块1: token 128..255  ← system 尾(128..136) + user(137..255)，累积哈希依赖 0..255 ❌
```

- 0~128 可保存（`prefix_tokens=137` 由 KCCM 自动 floor 到 128）；128~137 的尾部**无法独立保存**：只有满块才有哈希、复用单位是整块 slot。代价为每请求重复 prefill `prefix_len mod block_size` = 9 token，可忽略。
- 陷阱：把块 1 一并 pin 不等于 system prompt 级共享（其哈希含用户内容，仅对完全相同的 256 token 头命中）。**共享段结束位置必须落在块边界内**。
- 全覆盖方案：(a) 内容对齐——system prompt 补 padding 到块边界（bs=128 补到 256，bs=16 补到 144），生产通用做法；(b) 减小 block_size（尾巴 = prefix_len mod bs），代价是哈希/块表开销与 kernel 效率。HBM 与外部存储的复用粒度均由物理 block_size 决定（外部键可用 hash_block_size 细化匹配，但载荷仍块对齐）。

## 12. 附录：Agent 场景调用链走查

### 12.1 场景一：会话结束释放整个 session 的 KV

链路：session 内请求带 `cache_key=session-A`（共享 system prompt 单独 pin，pin_count=N）→ `POST /kv_cache/release {cache_key:"session-A"}` → API server → EngineCore RPC（DP 扇出，幂等）→ `KCCM.release` → 拆解为：私有块哈希注销 + 共享内容 `pin_count-1`（归零才注销）→ 下一调度步 patch 消费 ReleasePlan，`BlockPool` 移除命中注册，块留在 free 队列成为首要驱逐受害者。

语义澄清：

1. 减引用只作用于 pin 元数据；普通共享块无 per-session 所有权，session-A 的 release 不影响共享前缀对其他 session 的命中。
2. release != 物理回收：块本就在 free 队列可分配，release 的效果是"不可再命中 + 优先被驱逐"；物理复用发生在下次分配（经 `needs_kv_zeroing` 清零路径，隐私有保证）。

优化判定：低负载下几乎无物理效果；高负载下将死内容转为首选受害者，等效提升有效缓存容量，并立即归还 pin 配额。定位是容量治理 + 隐私 + 配额归还，非直接加速。

### 12.2 场景二：工具调用 10 分钟，TTL=600s

链路：最后一轮请求带 `mode=ttl, ttl_s=600, cache_key=session-A`（随 Request 原子到达，无竞态）→ 结束钩子捕获块哈希链建 TTL 条目 → 空闲期 `sweep_expired` 惰性检查不降级，A1 驱逐过滤保护这些块 → 工具返回，下一轮同前缀请求在 `get_computed_blocks` 命中 → `num_computed_tokens ≈ 会话长度`，只 prefill 新增 token；命中触发 touch 刷新 LRU，KCCM 以同 cache_key **替换/延长**条目（会话逐轮变长，覆盖新哈希链）。

优化判定：三个场景中唯一直接的推理加速（省整段会话重 prefill，改善 TTFT）。失效路径为软降级：晚于 TTL 返回则回落 NORMAL，可能部分命中或全量重算，正确性无损。

### 12.3 场景三：一次性大文档 no-store

链路：请求带 `mode=no_store` → `on_request_scheduled` 标记 → chunked prefill 各步 wrapped `cache_blocks` 跳过本请求新满块注册；AscendStore 开启时 store_mask 置空，不产生外部 put → 结束 free 后块匿名进入 free 队列，零残留。请求仍正常命中已有共享头（BOS/system），只是不写。

优化判定：开外部 KV Pool 时收益最大（省大文档 offload 带宽与外部容量）；纯 HBM 场景收益温和——LRU 天然偏向驱逐从未再命中的块，no-store 主要节省空闲窗口期死缓存占用、哈希表膨胀，并提供确定性隐私保证。

### 12.4 走查结论与设计修正

| 场景 | 优化性质 | 强度 |
| --- | --- | --- |
| 会话释放 | 容量治理 + 隐私 + 配额归还 | 中（高负载显现） |
| TTL | 直接省 TTFT（跨轮命中） | 强 |
| no-store | 外部存储 I/O + 死缓存治理 | 强（offload 开）/ 温和（纯 HBM） |

修正项：

1. 同 `cache_key` 后续 turn 结束时必须替换/延长条目而非新建。
2. release 语义显式拆为"私有注销 + pin_count 减"，API 文档须写明其非物理回收。
3. 场景一收益依赖负载压力，产品话术应定位为"清理与隔离"而非"腾内存加速"。

### 12.5 澄清：release 与 free 队列的关系（"清出 HBM"的正确定义）

需求更正为：release 后相关块应回到 free 空闲队列、随时可被重新分配。**该行为即 vLLM 默认生命周期，无需新机制**。块状态机：

```
运行中 (ref_cnt>0，不在 free 队列，不可分配)
   │ 请求结束 free()                    ← 此刻块即回队列，可随时重分配
   ▼
空闲+可命中 (在 free 队列，哈希已注册)  ← 新请求可随时取出复用；命中触发 touch 刷新位置
   │ release() = 仅注销哈希（变匿名，仍在队列）
   ▼
空闲+匿名 (在 free 队列，未注册)       ← 同样随时可分配，且是驱逐首选
   └──────────────→ 被新请求取出重新分配
```

要点：

1. 只有运行中请求持有的块不在队列；其余（可命中/匿名）全部随时可分配。"释放后可重分配"在请求 `free()` 时刻即成立，release 只是提前终止可命中状态并提升驱逐优先级。
2. pin/TTL 保护只影响受害者选择顺序（A1 过滤），从不把块移出队列，也不降低可分配性——"保护"与"可分配性"正交。
3. 曾考虑的"eager 立即清零"（数据即刻不可恢复）与该需求无关，降级为可选合规项：P0~P2 不实现，仅在确有"会话数据立即抹除"合规要求时，按"排除自身保护过滤 → 下一调度步注销+收集 block id → 按步预算分块 memset"方案 opt-in 启用。
4. 同机制换 scope 可支持条件化全量 flush：`POST /kv_cache/flush {"keep_protected": true}`（`reset_prefix_cache` 的保留保护版本）。

## 13. 业界参照实现调研

### 13.1 Claude Code（客户端侧参照，v2.1.88 反混淆源码）

Claude Code 是"纯客户端断点放置 + 服务端 TTL 淘汰"模型，客户端无法直接释放服务端缓存，只能提示。四个能力均有对应物：

**断点放置（soft pin 对应物）**：

- `getCacheControl()`（`source/src/services/api/claude.ts:358-374`）是唯一标记工厂：`{type:'ephemeral', ttl?:'1h', scope?:'global'|'org'}`。
- 消息历史恒 **1 个标记、逐轮滑动**（`addCacheBreakpoints`，claude.ts:3063-3106）：断点放最后一条消息，上一轮断点自然并入本轮缓存前缀，无"每 N 轮移动"启发式。注释透露服务端（内部代号 Mycro 的 page manager）按标记保护位置释放 local-attention KV 页——多余断点会让"永远不会再 resume"的位置多活一轮，故刻意只用 1 个。
- system prompt 按内容分块打标 ≤2 个（claude.ts:3213-237；`splitSysPromptPrefix` utils/api.ts:296-435）：静态段打标、动态段（attribution header、日期环境）**永不进缓存前缀**——"动态内容必须移出共享前缀"是贯穿性纪律。
- 工具区断点由服务端策略放置（`claude_code_system_cache_policy`），客户端职责是保证 built-in 工具为**连续排序前缀**（tools.ts:354-359），防止 MCP 工具插序打穿全部下游缓存键。
- 断点预算：API 上限 4，主循环实际用 2–3（yoloClassifier.ts:1094-1106 明确预算注释）。

**TTL**：默认 5m（隐式）；`ttl:'1h'` 由 GrowthBook allowlist（按 querySource 前缀匹配）× 用户资格**会话级锁存**（`should1hCacheTTL`，claude.ts:376-434，注释：中途翻转 ≈ 20K token 缓存重建）三路控制。TTL 感知行为：距上次 API >60min 判定缓存必然过期 → time-based microcompact 预裁剪（`timeBasedMCConfig.ts:3-28`，"反正要全量重写，先缩小重写量"）+ thinking 清除锁存。

**会话清理/主动释放（release 对应物）**：`tengu_cache_eviction_hint` 遥测事件（"Signal to inference that this conversation's cache can be evicted"），三个触发点：`conversation_clear`（/clear）、`session_end`（进程退出前 flush 保证不丢）、`subagent_end`，负载 `last_request_id` 定位缓存链。**只能提示、不能执行**——本设计的 KCCM 恰好是它的服务端执行者。

**no-store 对应物**：`skipCacheWrite`（forkedAgent.ts:110-112）——fire-and-forget fork 把断点从 `messages.length-1` 移到 `length-2`，即**只写与父对话的共享前缀，fork 自己的尾巴不写缓存**；使用方：压缩摘要 fork、侧问、离开摘要。配套纪律：MCP instructions、deferred tools、动态 agent 列表等变化内容全部移出缓存前缀（delta 附件化），注释记录"动态 agent 列表曾占 fleet cache_creation 的 10.2%"。

**会话稳定锁存（跨能力横切）**：TTL 资格锁存、tool schema 逐会话 memoize（toolSchemaCache.ts:3-8，"server position 2 的任何字节变化打穿 ~11K token 工具块"）、beta header 三锁存——全部为"会话中途不翻转缓存键"。

**超出本设计的能力**：cached microcompact（microCompact.ts:300-399）经 `cache_edits`/`cache_reference` 删除已缓存前缀中段的旧 tool_result 且剩余前缀继续命中，服务端返回 `cache_deleted_input_tokens`——说明 Anthropic 服务端支持**中段删除**；vLLM 内容寻址前缀缓存无此机制，本设计以 release+重新预热近似。

### 13.2 上游 vLLM（引擎侧参照，以本地 vllm/ 源码为准）

**驱逐链路（A1 patch 点确认）**：`FreeKVCacheBlockQueue` 为侵入式双向链表，队头=下一驱逐候选，LRU 语义（kv_cache_utils.py:165-215）；驱逐唯一主动点=分配时 `BlockPool.get_new_blocks` pop 队头 → `_maybe_evict_cached_block`（block_pool.py:333-400，摘哈希+`reset_hash`，物理块不动）；命中时 `touch` 把块摘出队列并 ref_cnt+1（:402-417）；释放顺序由 `free_blocks(ordered_blocks)` 决定未来驱逐序（:419-441，尾部块先逐）。**无任何优先级/保护位概念**，唯一保护是 `ref_cnt>0`。

**release 执行点可直接复用**：`BlockPool.evict_blocks(block_ids)`（block_pool.py:443-460）已实现按块 id 摘哈希并发出 `BlockRemoved` KV 事件（:392-399），上游仅用于 KV 加载失败路径（scheduler.py:2353-2399）。**本设计 release 无需新增 BlockPool API**，只需 hash→block_id 解析（经 `cached_block_hash_to_block`）。

**控制面通道（照抄即可）**：`call_utility("method_name", args)`（core_client.py:886-891）→ `EngineCoreRequestType.UTILITY`（b"\x03"）→ EngineCore 按方法名**反射**调用（core.py:1336-1369）→ Scheduler 同名方法。`reset_prefix_cache` 端到端范例：HTTP dev 路由（entrypoints/serve/dev/cache/api_router.py:20-43，仅 `VLLM_SERVER_DEV_MODE` 挂载）→ async_llm.py:921-924 → core.py:646-651 → scheduler.py:1943-1991 → `BlockPool.reset_prefix_cache`（全清哈希表）。四接口控制面照此模式新增方法。

**请求级控制面现状**：

- `cache_salt` 只进**第 0 块** extra_keys，后续块经 parent-hash 链传导隔离（kv_cache_utils.py:525-560）——namespace/cache_key 可直接落地为 salt，隔离免费获得。
- `skip_reading_prefix_cache`（sampling_params.py:342，消费于 kv_cache_manager.py:208-213）：**只有跳读、没有跳写**——no-store 确为上游空白，A2 patch 必要。
- 写缓存统一入口：`allocate_slots` → `coordinator.cache_blocks`（kv_cache_manager.py:422-434）——A2 wrapper 挂点。
- `kv_transfer_params`（request.py:101-117，经 `extra_args` 注入）是请求级 connector 控制的事实标准入口。

**其他可复用机制**：`delay_free_blocks`（scheduler.py:1888-1905，连接器请求结束后延迟释放，用于异步 offload）；NIXL 完整 per-request lease/TTL（nixl/scheduler.py:70-76 默认 30s/480s、worker 过期扫描 :1771-1790、心跳续租 :1847-1866）——外部层 TTL 的参照模式；OffloadingConnector 驱逐带 `protected` 保护集（cpu/manager.py:168-180，策略接口 `evict(n, protected)`）——KCCM 可直接向其注入保护哈希；`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`（envs.py:285，SWA 段尾全局保留）为本设计 per-content TTL 的退化特例。MooncakeStore 仅 `remove_all`、LMCache 仅按 request_id end_session，均无按内容删除。

### 13.3 Claude Code 概念 → 本设计映射

| Claude Code | 本设计对应 | 差异 |
| --- | --- | --- |
| `cache_control` ephemeral 断点 | pin（请求级声明） | CC 放标记位置，本设计按 cache_key+prefix_tokens 语义化 |
| 单标记逐轮滑动 | 请求随轮推进自动延长条目 | 无需显式移动断点 |
| `skipCacheWrite` | no-store（A2 patch） | 语义等价：只写共享前缀之外不写 |
| `tengu_cache_eviction_hint`（仅提示） | release（服务端执行） | KCCM 真正执行释放 |
| 5m/1h 服务端 TTL | `ttl_s` + sweep 降级 | 服务端自主控制，不依赖服务商标 |
| 4 断点预算 | pin 配额（比例 + namespace） | 从硬上限改为软配额 |
| 资格/schema/header 锁存 | 条目会话锁存（首次声明生效） | 防中途翻转缓存键 |
| `cache_edits` 中段删除 | 无对应（release+重预热近似） | vLLM 无 token 级编辑机制 |
| 动态内容移出前缀（delta 附件） | agent 侧纪律 + no-store | 引擎无法自动识别动态内容 |

### 13.4 对本设计的验证与修正

1. **release 执行简化**：复用 `BlockPool.evict_blocks` + `BlockRemoved` 事件（外部连接器可消费事件删键），原 A2"注销 patch"缩减为一个 hash→id 解析 helper。
2. **控制面定型**：四接口走 `call_utility` 反射通道，HTTP 路由参照 dev router 模式（转正需鉴权）。
3. **cache_key 落地为 cache_salt**：namespace 隔离免费获得，KCCM 条目以 salt 为 namespace 键。
4. **no-store 语义对标 skipCacheWrite**：只跳过"超出共享前缀的新满块"注册，与 `skip_reading_prefix_cache`（跳读）正交组合成读/写 2×2。
5. **会话锁存原则采纳**：请求级声明以首次生效，中途变更忽略（可另发控制面命令显式变更），防缓存键翻转。

## 14. 四接口落地实现方案

### 14.1 pin

```
Agent 请求级: kv_transfer_params {"kv_cache_control": {"mode":"pin","cache_key":"kb:doc:1","priority":10,"ttl_s":86400,"tier":"hbm"|"external"}}
Agent 控制面: POST /kv_cache/pin  → API server → call_utility("kv_cache_pin", ...) → EngineCore 反射 → scheduler.kv_cache_pin → KCCM.pin
生效链: on_request_scheduled 读声明 → on_request_finished 用 Request.block_hashes(request.py:175) 展开条目
        → A1 patch BlockPool.get_new_blocks: pop 受害者时跳过 protection_level>0 的块(重新入队尾=LRU刷新), O(跳过数/次)
配额: 全局比例(VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO) + namespace 上限; 超额 → 409 或降级 TTL
tier=external: 条目交给 offloading/连接器, 参照 OffloadingConnector protected 集注入
```

### 14.2 set_ttl

```
Agent 请求级: {"mode":"ttl","ttl_s":600,"cache_key":"session-A"}
控制面: POST /kv_cache/ttl → call_utility("kv_cache_ttl", ...) → KCCM.set_ttl
实现: LifecycleEntry.expire_at(monotonic); 每调度步开头 sweep_expired 惰性降级(NORMAL); 
      同 cache_key 后续 turn 结束 → 替换/延长条目(会话锁存: 首次声明生效, 控制面可显式变更)
external tier: 透传 TTL 给后端, 参照 NIXL lease 模式(per-request TTL + 过期扫描 + 心跳续租)
```

### 14.3 release

```
Agent: POST /kv_cache/release {cache_key} → call_utility("kv_cache_release", ...) → KCCM.release
执行链: 哈希过滤(排除自身保护) → patch helper BlockPool.evict_by_hashes(内部复用 evict_blocks, block_pool.py:443)
        → BlockRemoved KV 事件外发(:392-399) → 连接器消费删外部键(A3)
语义: 只注销+提升驱逐优先级; 块始终在 free 队列随时可分配(§12.5); pin_count-1 归零才注销共享条目
scope 扩展: {"keep_protected": true} 全量 flush = 条件化 reset_prefix_cache(上游 scheduler.py:1943-1991 同链路)
```

### 14.4 no-store

```
Agent 请求级: {"mode":"no_store"}（随 Request 原子到达, 无竞态; 控制面补救仅 best-effort）
实现: on_request_scheduled 登记 → patch KVCacheManager.cache_blocks(上游写点 kv_cache_manager.py:422-434,
      本仓调用点 recompute_scheduler.py:150 等) 对该请求跳过新满块注册; AscendStore store_mask 置空不 put
语义对标: skipCacheWrite —— 仍命中已有共享头, 只不写新内容; 与 skip_reading_prefix_cache 正交组成读/写 2×2
收尾: 请求结束 free 后块匿名入 free 队列, 零残留
```

### 14.5 实施文件清单（映射 §9 分期）

| 阶段 | 文件 | 内容 |
| --- | --- | --- |
| P0 | `vllm_ascend/core/kv_cache_control_manager.py`（新） | 元数据类+四接口+sweep+配额 |
| P0 | `vllm_ascend/patch/platform/patch_kv_cache_lifecycle.py`（新） | A2 cache_blocks wrapper + scheduler 生命周期钩子 + 声明解析 |
| P1 | `vllm_ascend/patch/platform/patch_kv_cache_eviction.py`（新） | A1 get_new_blocks 受害者过滤（上游漂移风险最高点，最小 patch 面） |
| P2 | EngineCore 方法 + API 路由 | `kv_cache_pin/ttl/release/flush` 走 call_utility；路由参照 dev router 加鉴权 |
| P2 | `pool_scheduler.py` 等 connector metadata | lifecycle 标记透传 + BlockRemoved 消费删外部键 |

UT 优先覆盖：A1 受害者选择矩阵（mock 上游 BlockPool，参照 `tests/ut/distributed/ascend_store/_mock_deps.py` 模式）、四接口幂等与配额、no-store 不留注册痕迹（prefix stats 验证）。

## 15. 实施计划（no-store 优先，其余接口仅定义）

### 15.1 关键设计决策（相对 §14 的简化）

1. **复用上游 `delay_cache_blocks`**：`KVCacheManager.allocate_slots`（vllm `kv_cache_manager.py:238-246`）已有 `delay_cache_blocks: bool = False` 参数（P/D 异步加载语义，`:422` 处跳过 `coordinator.cache_blocks`）。no-store 直接复用该语义，不自建写拦截。
2. **双 chokepoint 覆盖全部 scheduler**：写缓存入口有两条——`allocate_slots` 内部（`:434` 直接调 coordinator）与显式 `KVCacheManager.cache_blocks`（`:553`，recompute 等路径）。对这两处做 wrapper（`allocate_slots` 强制 `delay_cache_blocks=True`；`cache_blocks` no-op），**无需任何 scheduler 钩子**，自动覆盖 Recompute/ProfilingChunk/Balance/DynamicBatch/上游基类全部调度器。
3. **声明载体**：`kv_transfer_params["kv_cache_control"]`（OpenAI 层请求体顶层字段 `kv_transfer_params`，SDK 场景可用 `extra_body` 等价透传；request.py:101-117 已有通道），API 层零改动。
4. **惰性解析**：声明在 chokepoint 处按请求解析并 memoize 到 Request 的 duck attribute，热路径开销 = 1 次 dict 查找 + 1 次 getattr；no-store 无需跨请求状态，KVCM 无需清理逻辑。
5. KVCM 实例挂到 `KVCacheManager`（组合），不做模块级全局（符合仓库规范）。

### 15.2 KVCM 类骨架（P0）

```python
class KVCacheControlManager:
    """业务生命周期元数据管理。不持有 KV tensor，不直接修改 block。"""

    def __init__(self) -> None:
        self.metrics = {"no_store_requests": 0, "parse_errors": 0}

    @staticmethod
    def parse_request_control(request) -> dict | None:
        """解析 request.kv_transfer_params["kv_cache_control"]，规范化并 memoize。
        - 缺失/为空 → None；未知 mode → 忽略并 debug 日志（前向兼容）
        - 解析结果存 request 对象属性，二次调用直接返回"""

    def is_no_store(self, request) -> bool: ...

    # ---- 以下接口 P0 仅定义签名（§14），实现于 P1/P2 ----
    def pin(self, ref, *, priority: int = 0, ttl_s: float | None = None,
            tier: str = "hbm") -> PinHandle:
        raise NotImplementedError("P1")
    def set_ttl(self, ref, ttl_s: float | None) -> None:
        raise NotImplementedError("P1")
    def release(self, ref) -> bool:
        raise NotImplementedError("P2")
```

### 15.3 实施步骤

| # | 动作 | 文件 | 说明 |
| --- | --- | --- | --- |
| 1 | 新增 KVCM 类 | `vllm_ascend/core/kv_cache_control_manager.py`（新） | 纯逻辑，不 import 重型依赖，便于 UT |
| 2 | 环境变量 | `vllm_ascend/envs.py` | `VLLM_ASCEND_KV_CACHE_CONTROL`（默认 1：仅当请求带声明才生效，无声明零影响） |
| 3 | 双 wrapper | `vllm_ascend/patch/platform/patch_kv_cache_control.py`（新） | wrap `KVCacheManager.allocate_slots`（no-store → `delay_cache_blocks=True`）与 `KVCacheManager.cache_blocks`（no-store → no-op）；模块级 `_apply_patch()` 模式；幂等防重复 wrap |
| 4 | 注册 patch | `vllm_ascend/patch/platform/__init__.py` | 仿 `:52` 的 import 行 |
| 5 | 外部存储联动 | `pool_scheduler.py`（KVPoolScheduler） | 组装 store 请求元数据时检查声明，no-store → 不 put |
| 6 | 单元测试 | `tests/ut/core/test_kv_cache_control_manager.py`、`tests/ut/patch/test_kv_cache_control_no_store.py` | mock 上游依赖（参照 `_mock_deps.py` 模式） |
| 7 | 文档 | 设计文档状态更新 + 使用示例 | `kv_transfer_params` 示例 |

### 15.4 测试用例

**UT-1 声明解析（test_kv_cache_control_manager.py）**

| 用例 | 输入 | 期望 |
| --- | --- | --- |
| 正常 no-store | `{"kv_cache_control": {"mode": "no_store"}}` | 解析成功，`is_no_store=True`，metrics+1 |
| 无声明 | `kv_transfer_params=None` / 无 key | `is_no_store=False`，零开销路径 |
| 未知 mode | `{"mode": "future_mode"}` | 忽略（debug 日志），`False`，不抛异常 |
| 畸形载荷 | 非 dict / mode 非字符串 | `parse_errors+1`，`False`，不抛异常 |
| memoize | 同一请求两次解析 | 第二次不重新解析（用计数器/断言内部状态） |
| 存根契约 | 调 pin/set_ttl/release | `NotImplementedError`，签名与 §14 一致 |

**UT-2 双 wrapper 行为（test_kv_cache_control_no_store.py，mock 上游）**

| 用例 | 期望 |
| --- | --- |
| no-store 请求走 `allocate_slots` | 传给原方法的 `delay_cache_blocks=True`（用 mock 捕获参数）；coordinator.cache_blocks 未被调用 |
| no-store 请求走显式 `cache_blocks` | no-op（coordinator 未被调用） |
| 普通请求 | 原方法参数不变、正常调用（`delay_cache_blocks` 透传调用方原值） |
| 共享前缀不受影响 | 预注册的哈希表内容在 no-store 请求处理后不变（不误删他人注册） |
| `enable_caching=False` | 行为与基线一致 |
| patch 幂等 | 连续两次 apply 不产生双重 wrap |
| 开关关闭 | patch 不生效，行为与基线一致 |

**UT-3 外部存储联动**：no-store 请求的 store 元数据为空/put 被跳过（扩展现有 ascend_store UT）。

### 15.5 验收标准（容器 + NPU）

| # | 项 | 方法 | 通过标准 |
| --- | --- | --- | --- |
| A1 | no-store 功能生效 | 请求 A（携带 no_store，长且唯一的文档 P）→ 请求 B（相同 P，无声明）；对照组 A'（无声明）→ B'（相同 P） | B 的 `usage.prompt_tokens_details.cached_tokens ≈ 0`（仅公共系统前缀）；对照 B' 的 cached_tokens 显著 >0；服务端 `vllm:prefix_cache_hits` 指标一致 |
| A2 | 无声明零回归 | 不带声明跑现有 e2e 冒烟 + 前缀密集负载 | 输出正确性不变，prefix 命中率变化 <1% |
| A3 | 热路径开销 | 前缀密集负载 TPS/TTFT 前后对比 | 差异在噪声范围内（<1%） |
| A4 | 外部存储不写入（如启用 AscendStore） | no-store 请求后查询外部键 | 无该文档键 |
| A5 | 块正常回收 | no-store 大文档请求结束后并发其他流量 | 无内存水位异常，无 crash（块匿名入 free 队列可复用，§12.5） |

容器验证步骤：UT 全绿 → 单请求冒烟（带声明，输出正常）→ A1 A/B 对照 → A2/A3 回归 →（可选）A4。

### 15.6 风险提示

1. `delay_cache_blocks` 与 `cache_blocks` 签名依赖所 pin 的 vLLM 版本，升级时 wrapper 需跟随（锚点：`kv_cache_manager.py:238/:553`）。
2. wrapper 不得改变非 no-store 请求的任何参数；幂等保护必须做（patch 模块被多次 import 的场景）。
3. P/D disaggregated 场景下 `delay_cache_blocks` 原有语义（异步加载后补缓存）与 no-store 组合时的行为需在容器中专项验证一次。

## 16. 实现与验证记录（P0 no-store）

### 16.1 实现清单（分支 feat/kv-cache-control，基于 v0.23.0）

| 文件 | 状态 | 内容 |
| --- | --- | --- |
| `vllm_ascend/core/kv_cache_control_manager.py` | 新增 | KVCM 类：`parse_request_control`（解析+memoize 到 Request 属性）、`is_no_store`、metrics 计数、`ContentRef`/`PinHandle` 类型、`pin`/`set_ttl`/`release` 存根（NotImplementedError，P1/P2 实现） |
| `vllm_ascend/patch/platform/patch_kv_cache_control.py` | 新增 | 双 wrapper：`allocate_slots` 对 no-store 请求强制 `delay_cache_blocks=True`（含位置参数守卫：`args[5]` 替换）；`cache_blocks` 直接 no-op。`__vcc_no_store_patched__` marker 保证幂等；env 门控；模块级 `_apply_patch()` |
| `vllm_ascend/patch/platform/__init__.py` | 修改 | 注册 import（紧跟 `patch_kv_cache_coordinator`） |
| `vllm_ascend/envs.py` | 修改 | `VLLM_ASCEND_KV_CACHE_CONTROL`（默认 1；关闭后声明被忽略，行为与基线一致） |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | 修改 | KVPoolScheduler 4 个跳过点：`build_connector_meta` 两条 store 循环、`request_finished`、`request_finished_all_groups` 提前返回 `(False, None)` |
| `tests/ut/core/test_kv_cache_control_manager.py` | 新增 | UT-1 声明解析矩阵（13 用例） |
| `tests/ut/patch/platform/test_kv_cache_control_no_store.py` | 新增 | UT-2 wrapper 行为（9 用例：强制 delay、透传、位置参数守卫、cache_blocks no-op、幂等、env 开关） |
| `tests/ut/distributed/ascend_store/test_kvcc_no_store.py` | 新增 | UT-3 外部存储跳过（9 用例） |

关键实现说明：

1. 声明载体 `kv_transfer_params["kv_cache_control"]`（经请求体顶层字段 `kv_transfer_params` → serving 层写入 `extra_args["kv_transfer_params"]` → `Request.kv_transfer_params`，API 层零改动）。解析结果 memoize 到 Request 的 `_kv_cache_control_parsed` 属性，热路径开销为一次 dict 查找 + 一次 getattr。
2. 双 chokepoint 覆盖全部调度器（Recompute/ProfilingChunk/Balance/DynamicBatch/上游基类），无需任何 scheduler 钩子。
3. no-store 请求仍正常获得前缀命中（共享块不受影响），仅跳过新满块注册——对标 Claude Code `skipCacheWrite` 语义。
4. `pin`/`ttl` 已声明但未支持：解析时告警并忽略（前向兼容，`unsupported_requests` 计数）。

### 16.2 本地验证结果（无 torch/vllm 环境）

- UT-1 13 passed；UT-2 9 passed；UT-3 9 passed
- 回归：ascend_store 全套 183 passed + 10 skipped；`test_pool_scheduler.py` 68 passed
- `ruff check` / `ruff format` 通过

本地运行技巧（无 torch 环境）：`python -m pytest <file> --confcutdir=<test 所在目录> -p no:cacheprovider`，跳过 `tests/ut/conftest.py` 的真实 torch 依赖；UT-2 通过 `spec_from_file_location` 加载 patch 模块以绕过 `vllm_ascend/patch/platform/__init__.py` 的兄弟模块导入链。

### 16.3 容器验证指导

```bash
# 1) 单元测试（真实 vllm 环境，直接跑，无需 confcutdir）
pytest -sv tests/ut/core/test_kv_cache_control_manager.py \
  tests/ut/patch/platform/test_kv_cache_control_no_store.py \
  tests/ut/distributed/ascend_store/test_kvcc_no_store.py

# 2) A1 功能验收（A/B 对照）
# 请求 A：携带 no_store + 唯一长文档 P（建议 >2K token，多块）
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "<长文档 P>"}],
  "max_tokens": 8,
  "kv_transfer_params": {"kv_cache_control": {"mode": "no_store"}}
}'
# 请求 B：相同文档 P，不带声明
#   通过标准：B 的 usage.prompt_tokens_details.cached_tokens ~= 0（仅公共系统前缀）
# 对照组：A'（不带声明）+ B'（同文档）
#   通过标准：B' 的 cached_tokens 显著 > 0
# 交叉校验：curl -s http://127.0.0.1:8000/metrics | grep prefix_cache

# 3) A2/A3 回归：不带声明的常规流量，prefix 命中率与 TPS/TTFT 差异 < 1%
# 4) A4（启用 AscendStore 时）：no-store 请求结束后外部存储无该文档键
# 5) A5 稳定性：no-store 大文档请求后并发其他流量，内存水位正常、无 crash
```

说明：未配置 KVConnector 时 EngineCore 对 `kv_transfer_params` 打印 warning（`core.py` "Got kv_transfer_params, but no KVConnector found"），属预期，不影响本功能。

### 16.4 已知限制与后续

1. `pin`/`set_ttl`/`release` 为接口存根，按 §14 分期实现（P1/P2）。
2. P/D disaggregated 场景下 `delay_cache_blocks` 组合行为需专项验证一次。
3. 上游 vLLM 版本升级时，wrapper 锚点（`allocate_slots`/`cache_blocks` 签名）需复核。

### 16.5 容器端到端实测记录（2026-09-14）

本节记录 P0 no-store 在真实 Ascend NPU + 容器 + vLLM 0.23.0 环境下的完整 E2E 验证，覆盖 §15.5 验收标准 A1（功能）/ A2（零回归）/ A5（稳定性）；A4（外部存储不写入）需启用 AscendStore 连接器，不在本轮纯 HBM 验证范围内。

#### 16.5.1 环境说明

| 项 | 值 |
| --- | --- |
| 硬件 | Ascend 910B4 ×1（`ASCEND_VISIBLE_DEVICES=0`） |
| 容器运行时 | containerd + `nerdctl`（ascend OCI runtime） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.23.0`（vLLM 0.23.0，vllm-ascend editable install 于 `/vllm-workspace/vllm-ascend`） |
| 代码注入 | 将本分支 5 个改动文件 copy 覆盖进容器内 `/vllm-workspace/vllm-ascend`（editable install，纯 Python 无需重编译，避免整仓挂载遮蔽已编译 `.so`） |
| 模型 | Qwen/Qwen2.5-0.5B（HF 缓存快照 `060db6499f32faf8b98477b0a26969ef7d8b9987`） |
| 网络 | `--net host`（共享宿主网络，端口 8000） |

启动容器（宿主 shell）：

```bash
nerdctl run -d --name vllm-nostore-e2e --net host \
  --runtime /var/lib/npu-container-toolkit/runtime/ascend-docker-runtime \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/cxy/vllm-ascend:/src/vllm-ascend:ro \
  -v /home/llm_cache/huggingface:/root/.cache/huggingface:ro \
  quay.io/ascend/vllm-ascend:v0.23.0 sleep 300000
```

注入改动文件（5 个源码 + 3 个 UT）：

```bash
nerdctl exec vllm-nostore-e2e bash -lc '
SRC=/src/vllm-ascend; DST=/vllm-workspace/vllm-ascend
cp $SRC/vllm_ascend/core/kv_cache_control_manager.py              $DST/vllm_ascend/core/kv_cache_control_manager.py
cp $SRC/vllm_ascend/patch/platform/patch_kv_cache_control.py      $DST/vllm_ascend/patch/platform/patch_kv_cache_control.py
cp $SRC/vllm_ascend/patch/platform/__init__.py                    $DST/vllm_ascend/patch/platform/__init__.py
cp $SRC/vllm_ascend/envs.py                                       $DST/vllm_ascend/envs.py
cp $SRC/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
   $DST/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
'
```

启动推理服务（必须开启 `--enable-prompt-tokens-details`，否则 `cached_tokens` 恒为 `null`）：

```bash
nerdctl exec vllm-nostore-e2e bash -lc '
cd /workspace
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ASCEND_KV_CACHE_CONTROL=1
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
setsid nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL --served-model-name qwen2.5-0.5b \
  --host 0.0.0.0 --port 8000 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --max-model-len 8192 --dtype bfloat16 \
  > /tmp/vllm-serve.log 2>&1 &'
```

#### 16.5.2 测试脚本

脚本仅依赖标准库，保存为 `nostore_e2e.py` 后运行：

```python
#!/usr/bin/env python3
"""P0 no-store E2E 验收（A1 A/B 对照 + A2/A5 并发一致性 + metrics 交叉校验）。

前置：服务已就绪（16.5.1），并开启 --enable-prefix-caching
      --enable-prompt-tokens-details。仅依赖标准库。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8000"
MODEL = "qwen2.5-0.5b"


def make_doc(seed: str, blocks: int = 120) -> str:
    """生成一段唯一长文档（约 blocks*40 token，覆盖多个 KV block）。"""
    return " ".join(
        f"{seed} paragraph {i} the quick brown fox jumps over the lazy dog "
        f"cataloging distant galaxies and cryptographic protocols segment {i}."
        for i in range(blocks)
    )


def chat(doc: str, no_store: bool, max_tokens: int = 8) -> tuple[int, int]:
    """发送一次 chat 请求，返回 (cached_tokens, prompt_tokens)。

    no_store 时随请求声明 kv_transfer_params.kv_cache_control.mode=no_store。
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": doc}],
        "max_tokens": max_tokens,
    }
    if no_store:
        body["kv_transfer_params"] = {"kv_cache_control": {"mode": "no_store"}}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    usage = resp["usage"]
    details = usage.get("prompt_tokens_details") or {}  # 无命中时为 null
    return details.get("cached_tokens", 0), usage["prompt_tokens"]


def check(cond: bool, msg: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return cond


def main() -> int:
    ok = True

    print("== A1 no-store 功能（A/B 对照） ==")
    doc_x = make_doc("CONTROL-DOC-X")
    doc_y = make_doc("NOSTORE-DOC-Y")
    c_a1, _ = chat(doc_x, no_store=False)   # 对照 A'：首请求，无声明
    c_b1, _ = chat(doc_x, no_store=False)   # 对照 B'：同文档，无声明 → 应命中
    c_a2, _ = chat(doc_y, no_store=True)    # no-store A：长文档，带声明
    c_b2, _ = chat(doc_y, no_store=False)   # no-store B：同文档，无声明 → 应不命中
    print(f"  对照 A' cached={c_a1}  B' cached={c_b1}")
    print(f"  no-store A cached={c_a2}  B cached={c_b2}")
    ok &= check(c_b1 > 100 and c_b1 > c_a1, f"对照 B' 显著命中（cached={c_b1}）")
    ok &= check(c_b2 <= 1, f"no-store B 零命中（cached={c_b2}）")

    print("== A2/A5 无声明零回归 + 并发稳定性 ==")
    docs = [make_doc(f"BURST-{k}", blocks=40) for k in range(12)]
    flags = [False] * 6 + [True] * 6
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(chat, docs[i], flags[i]) for i in range(12)]
        for f in futs:
            f.result()  # 任一失败即抛异常
    print("  12 并发请求完成，无异常")
    consistent = True
    for i in range(12):
        rc, _ = chat(docs[i], no_store=False)  # 复播：普通应命中、no-store 不应
        consistent &= rc > 100 if i < 6 else rc == 0
    ok &= check(consistent, "普通请求重复命中 / no-store 请求重复不命中")

    print("== metrics 交叉校验（vllm:prefix_cache_*） ==")
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=30) as r:
        metrics = r.read().decode()

    def metric(name: str) -> float:
        for line in metrics.splitlines():
            if line.startswith(name):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    hits = metric("vllm:prefix_cache_hits_total")
    queries = metric("vllm:prefix_cache_queries_total")
    print(f"  prefix_cache_hits_total={hits}  queries_total={queries}")
    ok &= check(hits > 0 and queries > 0, "prefix cache 指标正常输出")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

#### 16.5.3 脚本使用说明

1. 按 16.5.1 启动容器、注入代码、拉起服务，确认 `curl http://127.0.0.1:8000/v1/models` 返回 200。
2. 保存 16.5.2 脚本为 `nostore_e2e.py`，用宿主 `python3` 运行（服务经 `--net host` 共享端口 8000）：
   ```bash
   python3 nostore_e2e.py
   ```
3. 通过标准：脚本逐项打印 `[PASS]`，末尾 `RESULT: PASS`，退出码 0。
4. UT 回归（真实 vllm 环境，容器内直接跑）：
   ```bash
   nerdctl exec vllm-nostore-e2e bash -lc 'cd /vllm-workspace/vllm-ascend && python3 -m pytest -q \
     tests/ut/core/test_kv_cache_control_manager.py \
     tests/ut/patch/platform/test_kv_cache_control_no_store.py \
     tests/ut/distributed/ascend_store/test_kvcc_no_store.py'
   ```

#### 16.5.4 测试用例与结果

| # | 验收项 | 用例 | 通过标准 | 实测结果 |
| --- | --- | --- | --- | --- |
| UT | 单元测试（真实 vllm 环境） | UT-1/2/3 共 31 例 | 全绿 | 31 passed |
| A1 | no-store 功能 | 对照 A'→B'（doc X，无声明）；no-store A→B（doc Y） | B' cached 显著 >0；B cached ≈0 | B'=4608，B=0 ✅ |
| A2 | 无声明零回归 | 6 普通请求复播 | 重复请求正常命中 | 复播 cached=640 ✅ |
| A5 | 块回收/并发稳定 | 12 并发（6 普通 + 6 no-store）复播 | 无异常、语义一致 | 12/12 无异常；no-store 复播 cached=0 ✅ |
| — | metrics 交叉校验 | 读 `/metrics` | `prefix_cache_hits`/`queries` 正常 | hits=4608，queries=19356 ✅ |

补充说明：

1. A1 对照组证实前缀缓存本身工作正常（B' 命中 4608 token），no-store 组证实声明后该文档**零注册**（B 命中 0 token）；二者区分度 100%，与 `vllm:prefix_cache_hits_total` 计数一致。
2. `"Got kv_transfer_params, but no KVConnector found. Disabling KVTransfer for this request."` 为预期 warning（未配置连接器），只影响外部 KV 池路径，不影响 HBM 前缀缓存的 no-store 语义（见 §16.3 说明）。
3. 未验证 A3（热路径开销 <1%）与 A4（外部存储不写入，需 AscendStore）——A3 需基准对比、A4 需连接器，均为后续工作。

## 17. P1/P2 实现与验收（pin / set_ttl / release / 控制面）

### 17.1 本轮决策

| 事项 | 决定 |
| --- | --- |
| 超额行为 | 控制面与请求级统一：降级为普通缓存 + warning + `quota_degraded` 计数，无 409 |
| 外部删除 | memcache 走 `batch_remove_lease` 真删；mooncake/yuanrong 无按 key 删除 API，打日志降级（要点标记） |
| HTTP 路由 | 本轮实现：wrap `api_server.build_app` 挂 `/kv_cache/*`，生产鉴权依赖部署网关（与 /reset_prefix_cache 同信任级） |
| 控制面 pin vs 请求级 pin | 请求级随请求原子声明（作用于本请求产出内容）；控制面作用于已有内容（依赖该 key 曾被请求级声明登记）；共享注册表，pin_count 叠加 |

### 17.2 实现清单

| 文件 | 内容 |
| --- | --- |
| `core/kv_cache_control_manager.py` | LifecycleEntry 注册表（(namespace, cache_key) 双索引：entries + hash→entries）、`pin`/`set_ttl`/`release`/`release_key`/`flush`/`protection_level`/`has_protection`/`maybe_sweep`/`on_request_finished`/`take_release_plan`、配额（`VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO`，默认 0.25，按 `len(hash_index)/num_gpu_blocks` 计）、惰性 sweep（挂 allocate_slots，next_expiry O(1) 快路径） |
| `patch/platform/patch_kv_cache_control.py` | 新增 free wrapper（finish 时捕获 `Request.block_hashes` 激活/替换条目，prefix_tokens 按 block_size floor）、allocate_slots 挂 sweep、init 绑定 block_pool 与 block_size |
| `patch/platform/patch_kv_cache_eviction.py`（新） | A1 驱逐过滤：`BlockPool.get_new_blocks` 逐块 pop，保护块（`protection_level>0`）暂存并 re-append 队尾（LRU 刷新）；**软 pin 兜底**：队列耗尽仍未收满则按队列顺序（最冷优先）牺牲保护块，永不阻塞分配，`pin_evictions` 计数；无保护 fast path |
| `patch/platform/patch_kv_cache_control_engine.py`（新） | EngineCore setattr 四方法（kv_cache_pin/set_ttl/release/flush，走 `call_utility` 反射通道）；AsyncLLM.`kv_cache_control_async`；wrap `build_app` 挂路由 |
| `entrypoints/kv_cache_router.py`（新） | `POST /kv_cache/{pin,ttl,release,flush}`，pydantic 请求体，async/sync client 兼容 |
| `ascend_store` 四文件 | `AscendConnectorMetadata.delete_keys` 字段；KVPoolScheduler.`queue_external_delete` + build_connector_meta 组装 `model@hash_hex` 键；AscendStoreConnector 代理；pool_worker.`get_finished` 消费 → `m_store.batch_remove_lease`（hasattr 探测 + 异常降级） |

release 语义：pin_count-1（handle）或清零（key）；归零注销并产出释放计划 = 条目哈希 − 仍被其他活跃条目保护的哈希（排除自身）；`_evict_hashes` 遍历 group 构造 `BlockHashWithGroupId` → `get_one_block` → `evict_blocks`（复用上游，自动发 `BlockRemoved` 事件）；块仅注销+提升驱逐优先级，不物理回收（§12.5）。TTL 到期 = 降级 NORMAL（`ttl_expired` 计数），非硬删。

### 17.3 测试与本地验证结果

| 文件 | 用例 |
| --- | --- |
| test_kv_cache_control_manager.py（14） | 解析矩阵 + pin 三态/TTL 过期/同 key 替换延长/pin_count 释放/释放计划排除他人保护/配额降级/prefix floor/sweep/flush |
| test_kv_cache_control_no_store.py（11） | 双 wrapper + free 钩子（捕获激活条目）+ 幂等 + env 开关 |
| test_kv_cache_eviction_filter.py（6） | fast path/保护跳过与 requeue/匿名块/全保护兜底/队列顺序/不足抛错 |
| test_kv_cache_control_engine.py（10） | EngineCore 转发/释放驱逐与外部删除队列/flush 两路/AsyncLLM 入口/build_app 挂载/路由端点 |
| test_kvcc_no_store.py（13） | no-store 跳过 + 外部删除组装/代理/worker 消费/不支持后端跳过 |

本地（无 torch/vllm）全部通过；ascend_store 全套 187 passed；ruff check/format 通过。

### 17.4 容器验收标准

| # | 场景 | 步骤 | 通过标准 |
| --- | --- | --- | --- |
| P1-1 | pin 抗驱逐 | 请求带 `{"mode":"pin","cache_key":"X"}` 产生长文档 → 灌多个大文档挤占 → 重复请求 X | X 的 cached_tokens 保持高位；未 pin 旧内容命中≈0 |
| P1-2 | TTL 过期 | `{"mode":"ttl","cache_key":"S","ttl_s":60}` | 30s 后请求命中>0；>60s+压力后命中≈0 |
| P1-3 | 同 key 延长 | 第二轮同 cache_key 声明 | 条目替换延长，metrics 单调 |
| P2-1 | release 隔离 | 会话 A release（共享提示词他方 pin） | A 私有前缀 miss；共享提示仍命中 |
| P2-2 | flush | `POST /kv_cache/flush {"keep_protected": true}` | 受保护保留，其余 miss |
| P2-3 | 超配额降级 | 调小 VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO 后 pin 大内容 | 行为等同普通请求，日志 warning，quota_degraded 递增 |
| P2-4 | 外部删除 | memcache 后端 release | `exists(keys)==0`；mooncake 后端日志降级 |
| 性能 | 驱逐过滤开销 | 高压+pin 场景 TPS/TTFT | 回归 <2%；无保护 fast path <1% |

curl 示例：

```bash
# pin（会话后对已有内容）
curl -s -X POST http://127.0.0.1:8000/kv_cache/pin -H "Content-Type: application/json" \
  -d '{"cache_key": "session-abc", "priority": 5, "ttl_s": 86400}'
# release（agent 会话结束钩子）
curl -s -X POST http://127.0.0.1:8000/kv_cache/release -H "Content-Type: application/json" \
  -d '{"cache_key": "session-abc"}'
# flush
curl -s -X POST http://127.0.0.1:8000/kv_cache/flush -H "Content-Type: application/json" -d '{"keep_protected": true}'
```

请求级声明（与 no-store 同通道）：`"kv_transfer_params": {"kv_cache_control": {"mode": "pin", "cache_key": "X", "priority": 5}}` 或 `{"mode": "ttl", "cache_key": "S", "ttl_s": 600}`。

### 17.5 已知限制与后续

1. namespace 级配额未实现（仅全局比例），多租户隔离依赖部署侧。
2. `tier="external"` 仅登记不降级（demote 复用 offload 体系，后续接入）。
3. 兜底牺牲保护块按队列顺序（LRU），未按 priority 细排；`pin_evictions` 指标可观测。
4. DP 场景控制面扇出与 `reset_prefix_cache` 同链路，多 DP 行为需容器专项确认。
5. 上游锚点：`BlockPool.get_new_blocks`/`evict_blocks`/`KVCacheManager.free`/`EngineCore` 反射，升级需复核。
