# zcode2api 适配 Antigravity CLI (agy) 工具调用排查与修复总结报告

**日期**: 2026-10-08  
**模块**: `app/gemini_compat.py`, `app/routes/gateway.py`  
**关联任务**: `A:\Paseo\HelloWorld\TASK_zcode2api_GLM_tools_fix.md`  

---

## 1. 现象复现与核心抓包定位

### 1.1 复现测试
在本地环境对 `http://127.0.0.1:3000` 执行测试请求：
1. **Gemini 流式端点 (`:streamGenerateContent?alt=sse`)**：
   收到单包响应：
   ```json
   data: {"candidates": [{"content": {"parts": [{"text": ""}], "role": "model"}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": null, "candidatesTokenCount": null, "totalTokenCount": null}, "modelVersion": "GLM-5.3-Flash"}
   ```
2. **Gemini 非流式 (`:generateContent`) / OpenAI 非流式 (`/v1/chat/completions`)**：
   直接返回 `502 Bad Gateway`，响应体为 `{"error":{"message":"上游响应格式异常","type":"upstream_error"}}`。

---

## 2. 根因深度剖析

### 2.1 诱因：上游智谱 Coding Plan 额度到期与“伪 200”穿透
- 通过直接请求上游抓包分析：
  当前本地 SQLite 数据库中的账号 `oauth-login`，其上游日额度窗口已于 `2026-10-07 12:00:00 UTC` 到期，当前 `billing/balance` 返回 `balances: []`。
- **致命特点**：上游智谱在额度超限/用尽时，HTTP 状态码返回 **`200 OK`**，但 Body 仅包含业务报错：
  ```json
  {"code": 1005, "msg": "exceed quota limit", "logid": "..."}
  ```
- **连锁反应**：
  1. **非流式解析**：网关只检查 `status_code >= 400`，对 200 判定为成功流；但在解析 JSON 时发现没有 `type: "message"`，直接粗暴抛出 `502 上游响应格式异常`。
  2. **流式解析**：因为该行 JSON 没有 `data:` 前缀，流式行读取器静默丢弃了该行；随后触发 `conv.finish_fallback()`，兜底吐出了单个带有 `STOP` 和空文本的包，导致 `agy` 客户端误判为模型结束回合。

### 2.2 代码缺陷：Gemini 协议转换层 ID 错配与 Null Usage
1. **Tool Use ID 严重错配**：
   - 在 `gemini_to_anthropic` 中，Gemini 模型输出的 `functionCall` 在转为 Anthropic 时合成了 `id: toolu_<uuid>`；
   - 但当客户端回传多轮对话中的 `functionResponse` 时，代码却将 `tool_use_id` 设置成了函数名字符串 `call.get("name")`（如 `"shell"`）；
   - Anthropic 上游严格要求 `tool_result` 的 `tool_use_id` 必须与上一轮 assistant 消息中的 `tool_use.id` 保持严格一致，否则直接拒绝请求。
2. **Usage 为 Null**：
   - `GeminiStreamConverter` 初始化时 `self.usage` 各字段全为 `None`，若上游未上报 prompt tokens，终态 chunk 直接将 `null` 序列化发向下游，引起严格解析器异常。
3. **报错信息黑盒**：
   - 网关层在解析非 Anthropic 格式的非流式响应时，未提取上游真实的 `msg`/`message` 字段，直接硬编码 `上游响应格式异常`，掩盖了配额用尽（`exceed quota limit`）的实际原因。

---

## 3. 修复实现与验证

### 3.1 代码修复清单
1. **`app/gemini_compat.py`**：
   - **ID 一致性配对**：将 `functionCall` 生成的 block `id` 与 `functionResponse` 的 `tool_use_id` 统一映射为 `call_{name}`，保证多轮对话上下文回传给上游时 ID 严格吻合。
   - **Usage 归零防 Null**：初始化 `usageMetadata` 默认值为 `0`，同步累加逻辑支持默认数值运算，杜绝下游接收到 `null`。
2. **`app/routes/gateway.py`**：
   - 优化非流式响应容错，若上游返回非标准格式（如包含 `code` 和 `msg` 的业务 JSON），提取其实际错误描述（例如 `exceed quota limit`），提升系统可观测性与排查效率。
3. **单元与集成测试**：
   - 更新 `tests/unit/test_gemini_compat.py` 中关于 tool_use 与 tool_result ID 配对的测试断言。
   - 全部 29 个 Gemini 单元与集成测试验证通过（`29 passed in 2.38s`）。

---

## 4. 上游配额现状说明
当前环境实际测试确认上游账号 `oauth-login` 处于额度耗尽状态（`code: 1005, exceed quota limit`），且上游目前暂未投放新活动套餐（`preview_plans` 为空）。  
后续如需继续通过 `agy` 进行端到端实机验证，只需通过 `python cli.py login zai` 或后台添加具备可用额度的账号即可顺畅执行工具调用。
