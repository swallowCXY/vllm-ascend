# KV Cache 生命周期控制 — 验收规格书

> 版本：v2.1（分支 `feat/kv-cache-control-v0.25.1rc1`，基线 vllm-ascend **v0.25.1rc1** / vLLM 0.25.1）
> 用途：与需求方确认功能范围、调用方式、验收标准与功能边界。
> 配套设计文档：`KV_Cache_Control_Manager.md`（§18 为本轮需求收敛后的最终实现记录）。
> 注：v0.23.0 基线的实测记录（设计文档 §16.5）为历史存档；本分支锚点已针对 vLLM 0.25.1 重新验证。

## 1. 范围与功能状态

| 功能 | 声明位置 | 状态 |
| --- | --- | --- |
| pin：前缀硬保护 + 必有 TTL（默认 1 小时） | message 内 | 已实现，待容器验收 |
| no-store：整个请求内容不写缓存 | message 内 | 已实现，P0 已实测通过 |
| release：整个请求缓存释放（会话收尾） | message 内 / HTTP | 已实现，待容器验收 |
| HTTP 控制面 | — | 仅保留 `POST /kv_cache/release`；pin/TTL/flush 不允许会话后操作 |

## 2. 语义基线（验收前必读）

1. **声明形态**：三个功能统一在 message 上声明 `kv_cache_control`，三 mode **互斥**——同请求出现 ≥2 种 mode 时全部不生效并返回提示。
2. **pin 范围**：保护 tools + 该消息之前的全部消息 + **声明消息本身**，尾部按块对齐截断（不完整块不保护）。一条请求最多一个 pin message。
3. **pin 硬性不淘汰**：保护块绝不参与驱逐。pin 池受配额限制（占总 KV 块比例），超限时新 pin 不激活、正常推理、日志提示。
4. **pin 必有 TTL**：缺省 3600s（`VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S`），显式 `ttl_s` 覆盖；到期解除保护（非定时删除），内容由 LRU 决定回收。
5. **pin 生效时机**：自请求**完成时**激活（请求进行中的块不受保护）。
6. **release 全删语义**：注销该请求块哈希覆盖的**全部**缓存注册——包括与其他会话共享且未被对方 pin 的前缀（对方将 miss 重算）。这是简化设计的明确代价。
7. **非物理回收**：release/TTL 到期只注销命中注册；块立即回到可分配队列，数据在新块覆写时消失。
8. **块粒度**：所有边界按块对齐；边界计算依赖 chat template 前缀单调性，不满足时该声明不生效并返回提示（不报错）。

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
| 其余字段 | — | 不支持（如 priority/tier/cache_key 已移除） |

Python SDK（经 `extra_body` 无法按 message 传递非标字段时，直接使用 HTTP JSON）：

```python
import requests
requests.post(f"{BASE}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [
        {"role": "user", "content": doc,
         "kv_cache_control": {"mode": "pin"}},
    ],
})
```

### 3.2 HTTP 控制面（仅 release）

```
POST /kv_cache/release
{"request_id": "chatcmpl-..."}          → {"released": true|false}
```

- 按已完成请求的 `request_id` 注销其全部缓存注册
- 依赖引擎内的请求记录表（容量 `VLLM_ASCEND_KVCC_RELEASE_TABLE_SIZE`=4096，FIFO 淘汰；超出容量的旧请求返回 `released=false`）
- 未配置 KVConnector 时引擎对 `kv_transfer_params` 打印 warning，属预期

### 3.3 配置项

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `VLLM_ASCEND_KV_CACHE_CONTROL` | `1` | 总开关；`0` 时所有声明被忽略，行为与基线一致 |
| `VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO` | `0.25` | pin 保护块数占总 GPU KV 块上限比例；超限新 pin 不激活 + 日志 |
| `VLLM_ASCEND_KVCC_DEFAULT_PIN_TTL_S` | `3600` | pin 默认 TTL |
| `VLLM_ASCEND_KVCC_RELEASE_TABLE_SIZE` | `4096` | HTTP release 可追溯的已完成请求数 |

### 3.4 观测通道

| 通道 | 用途 |
| --- | --- |
| 响应 `usage.prompt_tokens_details.cached_tokens`（需 `--enable-prompt-tokens-details`） | 命中判定主依据（>100 命中 / ≤1 未命中） |
| 非流式响应字段 `kv_cache_control_status` | 声明裁决结果：`{"status": "accepted"/"rejected", "reason": ...}`；rejected 原因：`conflict_modes` / `multiple_pin_messages` / `invalid_declaration` / `boundary_render_failed` |
| `GET /metrics` 的 `vllm:prefix_cache_hits/queries` | 交叉校验 |
| 服务端日志 | 配额超限（pin 超预算不激活）、TTL 过期等事件 |

## 4. 验收规格与用例

启动要求：容器 + NPU（参照设计文档 §16.5.1），参数 `--enable-prefix-caching --enable-prompt-tokens-details`。判据统一：`cached_tokens > 100` 命中、`≤ 1` 未命中。

### 4.1 no-store（AC-NS，声明于 message 内）

| ID | 用例 | 步骤 | 通过标准 |
| --- | --- | --- | --- |
| AC-NS-01 | 不写缓存 | 请求 A：任一 message 带 `{"mode":"no_store"}` + 唯一长文档 P → 请求 B（同 P，无声明） | B cached_tokens ≤ 1 |
| AC-NS-02 | 对照命中 | A'（同 P 无声明）→ B'（同 P 无声明） | B' cached_tokens > 100 |
| AC-NS-03 | 读命中保留 | no-store 请求与已有缓存共享系统前缀 | 共享前缀仍命中 |
| AC-NS-04 | 并发回归 | 混合声明/无声明并发后逐个无声明重放 | 声明内容 miss、普通内容命中、无异常 |

### 4.2 pin（AC-PIN）

| ID | 用例 | 步骤 | 通过标准 |
| --- | --- | --- | --- |
| AC-PIN-01 | 硬保护抗驱逐 | message 带 `{"mode":"pin"}` 产生长前缀（含 tools）→ 灌多个大文档挤占 → 重放同前缀 | cached_tokens > 100（未被逐）；未 pin 旧内容 ≤ 1 |
| AC-PIN-02 | 默认 TTL | pin 不带 ttl_s → 1h 内重放 | 命中 |
| AC-PIN-03 | 显式 TTL 过期 | pin `ttl_s=60` → 60s 内重放命中 → >60s + 驱逐压力 → 重放 | 先命中后 miss |
| AC-PIN-04 | 保护范围含 tools/前缀消息 | pin 最后一条消息，前面还有普通消息 | 前缀整体命中（边界覆盖前面消息与 tools） |
| AC-PIN-05 | 超配额降级 | `VLLM_ASCEND_KVCC_PIN_BUDGET_RATIO=0.001` 重启后 pin 大前缀 | 行为等同普通内容（压力后 miss）+ 日志 warning + `quota_degraded` |
| AC-PIN-06 | 响应提示 | 带 pin 的非流式请求 | `kv_cache_control_status == {"status":"accepted","reason":"pin"}` |

### 4.3 release（AC-REL）

| ID | 用例 | 步骤 | 通过标准 |
| --- | --- | --- | --- |
| AC-REL-01 | 请求级释放 | sub agent 收尾请求带 `{"mode":"release"}` → 完成后重放同前缀 | cached_tokens ≤ 1（确定性） |
| AC-REL-02 | HTTP 释放 | 请求完成后 `POST /kv_cache/release {"request_id"}` → 重放 | miss；响应 `released=true` |
| AC-REL-03 | 幂等/未登记 | 重复 release / 未知 request_id | `released=false`，无异常 |
| AC-REL-04 | 全删语义确认 | release 的请求命中过共享前缀 S（S 未被 pin）→ 释放后重放 | S 也 miss（全删语义，需求方需认可） |

### 4.4 互斥（AC-CONF）

| ID | 用例 | 通过标准 |
| --- | --- | --- |
| AC-CONF-01 | pin + no_store 混合 | 两者均不生效：内容可写缓存且无保护；响应 `status=rejected, reason=conflict_modes` |
| AC-CONF-02 | pin + release / release + no_store 混合 | 同上，全部不生效 + 提示 |
| AC-CONF-03 | 多个 pin message | rejected，`reason=multiple_pin_messages` |

### 4.5 性能与稳定性

| ID | 用例 | 通过标准 |
| --- | --- | --- |
| AC-REG-01 | 无声明流量回归 | 输出正确；命中率变化 < 1% |
| AC-PERF-01 | 热路径开销 | TPS/TTFT 差异 < 1%（无声明） |
| AC-PERF-02 | 驱逐过滤开销 | 高压 + pin 场景 TPS/TTFT 差异 < 2% |
| AC-STAB-01 | 长稳 30min | 无 crash、无内存异常、pin 内容持续命中 |

## 5. 功能边界（明确不保证的行为）

1. **pin 硬保护 ≠ 无限保护**：受 TTL 约束（默认 1h）；自请求完成起生效；极端压力下系统可能对非 pin 流量产生抢占（pin 池最大占配额比例）。
2. **release 全删**：不做"仅删本请求新增"的区分——共享且未 pin 的前缀会被连带注销（AC-REL-04）。被覆盖的他人 pin 条目残留但失去保护对象；同内容被重新计算注册后，原 TTL 内会再次受保护。
3. **非物理回收/无即时清零**：仅注销命中注册；需要"数据立即不可恢复"合规语义需另行立项。
4. **块粒度与边界最佳努力**：尾部不完整块不保护；模板非前缀单调时声明不生效并提示。
5. **互斥即全不生效**：混合 mode 的整条声明丢弃（含合法部分），以响应提示告知。
6. **配额超限无同步提示**：配额判定在请求完成时（引擎侧），仅日志 + 计数；响应中的 status 只反映 serving 层可判定的冲突类拒绝。
7. **流式响应无 status 字段**：仅日志（非流式有 JSON 字段）。
8. **控制面仅 release**：pin/TTL/flush 不可会话后操作；release 按 request_id 受记录表容量限制；无内置鉴权（依赖部署网关）。
9. **无 external store 集成**：tier/外部删除已移除。
10. **无 Prometheus 导出**：生命周期计数未接入 /metrics。
11. **易失性**：注册表/记录表为进程内存态，重启失效。
12. **模型范围**：标准 prefix caching 模型（FullAttention 族）；hybrid/SWA 未验证。
13. **上游锚点**：依赖 vLLM 0.23.0（`delay_cache_blocks`/`evict_blocks`/utility 反射/serving 结构），升级需回归。

## 6. 验收结果记录

| 用例 ID | 结果 | 执行人 | 日期 | 备注 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## 7. 参考

- 设计文档 `KV_Cache_Control_Manager.md`（§18 最终实现记录；§15/§16 no-store 历史；§12 语义附录）
- P0 E2E 脚本 `nostore_e2e.py`（设计文档 §16.5.2）可扩展为本文 §4 全量验收脚本
