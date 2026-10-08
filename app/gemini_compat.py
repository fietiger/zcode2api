"""Gemini generateContent 兼容层 —— /v1beta/models/*:generateContent ↔ Anthropic Messages 双向转换。

入站：Gemini 请求体（contents 角色 user/model、systemInstruction、tools.functionDeclarations、
functionCall / functionResponse 回合）→ Anthropic messages 体。
出站：Anthropic 响应（JSON 或 SSE 事件流）→ Gemini GenerateContentResponse JSON / SSE chunk 流。

协议实证（agy CLI 网关模式抓包，2026-10）：
- 角色 enum 只有 user / model；system prompt 在 systemInstruction 与 contents 分离
- 工具调用输出为 parts 中的 {"functionCall": {"name", "args"}}；客户端回传
  {"functionResponse": {"name", "response"}}——Gemini 协议不带调用 id，靠 name 关联
- thinking 内容输出为 parts: [{"thought": true, "text": "..."}]（includeThoughts）
- agy 收到空 parts 会注入「model output must contain either output text or tool calls」
  错误并无限重试——所有产出点必须保证 parts 非空才输出

与 openai_compat 同一原则：对畸形输入保持宽容，映射不了的部件安静跳过，
绝不放大请求失败。
"""

from __future__ import annotations

import json
import uuid

# Anthropic stop_reason → Gemini finishReason（agy 按 STOP/MAX_TOKENS 语义分支）
_STOP_REASON_MAP = {
    "end_turn": "STOP",
    "stop_sequence": "STOP",
    "max_tokens": "MAX_TOKENS",
    "tool_use": "STOP",
}

# Gemini 错误 status（google.rpc.Code 文本形态，agy 按 status 决定重试策略）
_ERROR_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    502: "INTERNAL",
    503: "UNAVAILABLE",
}


def gemini_error(status_code: int, message: str) -> dict:
    """统一 Gemini 错误体（google.api.http error body 形态）。"""
    return {"error": {"code": status_code, "message": message,
                      "status": _ERROR_STATUS.get(status_code, "INTERNAL")}}


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _inline_image_block(part: dict) -> dict | None:
    """inlineData（base64 原文）→ Anthropic image block；缺 mime/data 跳过。"""
    data = part.get("inlineData") or part.get("inline_data")
    if not isinstance(data, dict):
        return None
    mime = str(data.get("mimeType") or data.get("mime_type") or "image/png")
    b64 = data.get("data")
    if not isinstance(b64, str) or not b64:
        return None
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}


def _blocks_from_parts(parts: object) -> list[dict]:
    """Gemini parts → Anthropic content blocks。"""
    blocks: list[dict] = []
    if not isinstance(parts, list):
        return [{"type": "text", "text": ""}]
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
        elif isinstance(part.get("functionCall"), dict):
            call = part["functionCall"]
            args = call.get("args")
            fn_name = str(call.get("name") or "")
            blocks.append({
                "type": "tool_use",
                "id": f"call_{fn_name}",
                "name": fn_name,
                "input": args if isinstance(args, dict) else {},
            })
        elif isinstance(part.get("functionResponse"), dict):
            fr = part["functionResponse"]
            response = fr.get("response")
            if isinstance(response, (dict, list)):
                content = json.dumps(response, ensure_ascii=False)
            elif isinstance(response, str):
                content = response
            else:
                content = "" if response is None else str(response)
            fn_name = str(fr.get("name") or "")
            blocks.append({
                "type": "tool_result",
                "tool_use_id": f"call_{fn_name}",
                "content": content,
            })
        else:
            block = _inline_image_block(part)
            if block:
                blocks.append(block)
    return blocks or [{"type": "text", "text": ""}]


def _instruction_texts(si: object) -> list[str]:
    """systemInstruction（{parts:[...]} | [parts] | str 三形态）→ 文本列表。"""
    if isinstance(si, str):
        return [si] if si.strip() else []
    if not isinstance(si, dict) and not isinstance(si, list):
        return []
    items = si.get("parts") if isinstance(si, dict) else si
    out: list[str] = []
    for part in items or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
            out.append(part["text"])
    return out


def gemini_to_anthropic(payload: dict, model: str | None = None) -> tuple[dict | None, str | None]:
    """Gemini 请求体 → Anthropic messages 体。非法时返回 (None, 错误信息)。

    model：路径段解析出的模型名（/v1beta/models/{model}:generateContent），
    优先级低于 payload 内显式 model 字段。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    system_parts = _instruction_texts(payload.get("systemInstruction") or payload.get("system_instruction"))

    out_msgs: list[dict] = []

    def _append(role: str, blocks: list[dict]) -> None:
        # 主 agent 循环会产生 model(functionCall) → model(functionResponse 回传前的
        # 中间文本) 等连续同角色段；Gemini 逐 content 发，Anthropic 合并进同一条
        if out_msgs and out_msgs[-1]["role"] == role:
            out_msgs[-1]["content"].extend(blocks)
        else:
            out_msgs.append({"role": role, "content": list(blocks)})

    for content in payload.get("contents") or []:
        if not isinstance(content, dict):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        _append(role, _blocks_from_parts(content.get("parts")))

    if not out_msgs:
        out_msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})

    gc = payload.get("generationConfig") if isinstance(payload.get("generationConfig"), dict) else {}
    # agy 不带 maxOutputTokens：给足上限内的默认值（上游侧由 _normalize_body 再钳制）
    max_tokens = _as_int(gc.get("maxOutputTokens")) or 32768

    body: dict = {
        "model": str(payload.get("model") or model or ""),
        "messages": out_msgs,
        "max_tokens": max_tokens,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    try:
        if gc.get("temperature") is not None:
            body["temperature"] = float(gc["temperature"])
        if gc.get("topP") is not None:
            body["top_p"] = float(gc["topP"])
        stops = gc.get("stopSequences")
        if isinstance(stops, list) and stops:
            body["stop_sequences"] = [str(s) for s in stops if s]
    except (TypeError, ValueError):
        pass

    for t in payload.get("tools") or []:
        if not isinstance(t, dict):
            continue
        mapped = []
        for fn in t.get("functionDeclarations") or []:
            if isinstance(fn, dict) and fn.get("name"):
                mapped.append({
                    "name": str(fn["name"]),
                    "description": str(fn.get("description") or ""),
                    # parameters 即 JSON Schema，直接作 input_schema
                    "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"},
                })
        if mapped:
            body.setdefault("tools", []).extend(mapped)

    mode = None
    if isinstance(payload.get("toolConfig"), dict):
        cfg = payload["toolConfig"].get("functionCallingConfig")
        if isinstance(cfg, dict):
            mode = cfg.get("mode")
    if mode == "NONE":
        body.pop("tools", None)
    elif mode == "ANY":
        body["tool_choice"] = {"type": "any"}
    # AUTO / 缺省：Anthropic 默认即 auto；thinkingConfig 纯 Gemini 侧开关，忽略

    return body, None


def _usage_from(usage: dict) -> dict:
    in_tok = _as_int(usage.get("input_tokens")) or 0
    out_tok = _as_int(usage.get("output_tokens")) or 0
    return {"promptTokenCount": in_tok, "candidatesTokenCount": out_tok, "totalTokenCount": in_tok + out_tok}


def _parts_from_blocks(blocks: object) -> list[dict]:
    """Anthropic content blocks → Gemini parts（跳过空块，保证非空语义）。"""
    parts: list[dict] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"]:
            parts.append({"text": block["text"]})
        elif block.get("type") == "tool_use":
            args = block.get("input")
            parts.append({"functionCall": {
                "name": str(block.get("name") or ""),
                "args": args if isinstance(args, dict) else {},
            }})
    return parts


def anthropic_to_gemini(data: dict, model: str) -> dict:
    """Anthropic message 响应 → 单个 GenerateContentResponse JSON（非流式）。"""
    parts = _parts_from_blocks(data.get("content"))
    if not parts:
        # 空 parts 会触发 agy「必须含输出文本或工具调用」错误循环——兜底占位
        parts = [{"text": ""}]
    return {
        "candidates": [{
            "content": {"parts": parts, "role": "model"},
            "finishReason": _STOP_REASON_MAP.get(data.get("stop_reason"), "STOP"),
            "index": 0,
        }],
        "usageMetadata": _usage_from(data.get("usage") or {}),
        "modelVersion": str(data.get("model") or model),
    }


class GeminiStreamConverter:
    """Anthropic SSE 事件流 → Gemini GenerateContentResponse SSE chunk 流（有状态转换器）。

    块生命周期映射：
      text 块：text_delta 逐片透出 {"text": ...} part
      thinking 块：同形透出 {"thought": true, "text": ...} part（agy includeThoughts）
      tool_use 块：content_block_start 起缓存 + input_json_delta 累积参数，
        content_block_stop 时整体拼 {"functionCall": {name, args}} 一次输出
      message_delta：发含 finishReason + usageMetadata 的终态 chunk
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.finish_reason: str | None = None
        # 默认有效整数（避免传 null，对齐 agy/Gemini 协议客户端解析规范）
        self.usage = {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
        self._tool: dict | None = None   # 进行中的 tool_use 块（累积 args json 片段）
        self._terminal_emitted = False   # 是否已发出 finishReason 终态 chunk
        self._started = False            # 是否收到过 message_start（有真实进度）

    def _sync_usage_total(self) -> None:
        p = self.usage.get("promptTokenCount") or 0
        c = self.usage.get("candidatesTokenCount") or 0
        self.usage["totalTokenCount"] = p + c

    def _chunk(self, parts: list[dict]) -> str:
        return f"data: {json.dumps({'candidates': [{'content': {'parts': parts, 'role': 'model'}, 'index': 0}]}, ensure_ascii=False)}\n\n"

    def _final_chunk(self, finish_reason: str) -> str:
        # 终态 chunk：finishReason 承载在 candidate 上，parts 给空文本占位
        #（Gemini 官方流里终态 chunk 的 content.parts 可为占位文本，客户端只读 finishReason）
        payload = {
            "candidates": [{
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": finish_reason,
                "index": 0,
            }],
            "usageMetadata": dict(self.usage),
            "modelVersion": self.model,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def start(self) -> str:
        return ""  # Gemini SSE 无 role 首 chunk，直接等内容

    def done(self) -> str:
        return ""  # 无 [DONE] 哨兵，流关闭即结束

    def finish_fallback(self, *, failed: bool = False) -> list[str]:
        """流收尾兜底：仅在尚未发出终态 chunk 时补发，幂等。

        截断/异常时补一个 STOP 终态 chunk（带上游已知的 usage），保证 agy 收到
        终态而不是挂等或空流重试。
        """
        if self._terminal_emitted:
            return []
        self._terminal_emitted = True
        return [self._final_chunk("STOP")]

    def feed(self, evt: dict) -> list[str]:
        etype = evt.get("type")
        if etype == "message_start":
            self._started = True
            u = (evt.get("message") or {}).get("usage") or {}
            prompt = _as_int(u.get("input_tokens"))
            if prompt is not None:
                self.usage["promptTokenCount"] = prompt
                self._sync_usage_total()
            return []
        if etype == "content_block_start":
            block = evt.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._tool = {"name": str(block.get("name") or ""), "args": ""}
            return []
        if etype == "content_block_delta":
            delta = evt.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"]:
                # thought 标志由 content_block_start 的块类型决定；text_delta 本身
                # 不带类型，thinking 块在 Anthropic 侧是独立 index 的 thinking_delta
                return [self._chunk([{"text": delta["text"]}])]
            if delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str) and delta["thinking"]:
                return [self._chunk([{"thought": True, "text": delta["thinking"]}])]
            if delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                if self._tool is not None:
                    self._tool["args"] += delta["partial_json"]
            return []
        if etype == "content_block_stop":
            if self._tool is None:
                return []
            tool, self._tool = self._tool, None
            try:
                args = json.loads(tool["args"]) if tool["args"] else {}
            except ValueError:
                args = {"_raw": tool["args"]}
            if not isinstance(args, dict):
                args = {"_raw": tool["args"]}
            return [self._chunk([{"functionCall": {"name": tool["name"], "args": args}}])]
        if etype == "message_delta":
            delta = evt.get("delta") or {}
            out = _as_int((evt.get("usage") or {}).get("output_tokens"))
            if out is not None:
                self.usage["candidatesTokenCount"] = out
                self._sync_usage_total()
            self._terminal_emitted = True
            self.finish_reason = _STOP_REASON_MAP.get(delta.get("stop_reason"), "STOP")
            return [self._final_chunk(self.finish_reason)]
        if etype == "message_stop":
            # message_delta 已发终态 chunk；若上游截断（无 message_delta），
            # finish_fallback 会补——此处只标记，不再产出
            self._terminal_emitted = True
            return []
        return []  # ping / error / content_block_start(空) 等无需产出
