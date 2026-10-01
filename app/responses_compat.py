"""OpenAI Responses API 兼容层 —— /v1/responses ↔ Anthropic Messages 双向转换。

入站：Responses 请求体（input 字符串或 item 数组、instructions、tools 扁平
函数定义、function_call / function_call_output 历史项）→ Anthropic messages 体。
出站：Anthropic 响应（JSON 或 SSE 事件流）→ Responses 响应对象 / 事件流
（response.created、output_item.added/done、output_text.delta、
function_call_arguments.delta、response.completed …）。

与 openai_compat 同一原则：对畸形输入保持宽容，映射不了的部件（reasoning
项、item_reference 等）安静跳过，绝不放大请求失败。
"""

from __future__ import annotations

import json
import time
import uuid

_STOP_STATUS_MAP = {
    "end_turn": "completed",
    "stop_sequence": "completed",
    "max_tokens": "incomplete",
    "tool_use": "completed",
}


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _text_from_parts(parts: list[dict]) -> str:
    return "\n".join(p for p in parts if p)


def _blocks_from_content_parts(content: object) -> list[dict]:
    """Responses content 分块（input_text/output_text/input_image）→ Anthropic blocks。"""
    blocks: list[dict] = []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else [{"type": "text", "text": ""}]
    if not isinstance(content, list):
        return [{"type": "text", "text": ""}]
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text") and isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
        elif ptype in ("input_image", "image_url"):
            url = part.get("image_url") if isinstance(part.get("image_url"), str) else (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                head, _, b64 = url.partition(",")
                media_type = head[5:].split(";")[0] or "image/png"
                if b64:
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
    return blocks or [{"type": "text", "text": ""}]


def responses_to_anthropic(payload: dict) -> tuple[dict | None, str | None]:
    """Responses 请求体 → Anthropic messages 体。非法时返回 (None, 错误信息)。"""
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return None, "必须提供 model 参数"

    system_parts: list[str] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_parts.append(instructions)

    out_msgs: list[dict] = []

    def _append_text(role: str, blocks: list[dict]) -> None:
        if out_msgs and out_msgs[-1]["role"] == role:
            out_msgs[-1]["content"].extend(blocks)
        else:
            out_msgs.append({"role": role, "content": list(blocks)})

    raw_input = payload.get("input")
    items: list[object]
    if isinstance(raw_input, str):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": raw_input}]}]
    elif isinstance(raw_input, list):
        items = raw_input
    else:
        return None, "input 必须是字符串或数组"

    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message":
            role = item.get("role")
            blocks = _blocks_from_content_parts(item.get("content"))
            if role == "assistant":
                # Anthropic 侧 assistant 文本直接并块；空文本补空 text 占位
                _append_text("assistant", blocks or [{"type": "text", "text": ""}])
            else:
                _append_text("user", blocks)
        elif itype == "function_call":
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            if args is None:
                args = {}
            elif not isinstance(args, dict):
                args = {"_raw": args}
            _append_text("assistant", [{
                "type": "tool_use",
                "id": str(item.get("call_id") or item.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"),
                "name": str(item.get("name") or ""),
                "input": args,
            }])
        elif itype == "function_call_output":
            output = item.get("output")
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False)
            elif not isinstance(output, str):
                output = "" if output is None else str(output)
            # Responses 的 output 可能是 {"output": "..."} 包装
            try:
                wrapped = json.loads(output)
                if isinstance(wrapped, dict) and "output" in wrapped and set(wrapped.keys()) <= {"output", "metadata"}:
                    inner = wrapped["output"]
                    output = inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
            except ValueError:
                pass
            _append_text("user", [{
                "type": "tool_result",
                "tool_use_id": str(item.get("call_id") or ""),
                "content": output,
            }])
        # reasoning / item_reference / web_search_call 等无法回填的类型安静跳过
        # （连续 assistant 文本/工具块由 _append_text 自动合并进同一条消息）

    # Anthropic 要求消息角色交替可由上游宽容处理，但至少要有一条消息
    if not out_msgs:
        out_msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})

    body: dict = {
        "model": model,
        "messages": out_msgs,
        "max_tokens": _as_int(payload.get("max_output_tokens")) or 4096,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    try:
        if payload.get("temperature") is not None:
            body["temperature"] = float(payload["temperature"])
        if payload.get("top_p") is not None:
            body["top_p"] = float(payload["top_p"])
    except (TypeError, ValueError):
        pass
    if payload.get("stream"):
        body["stream"] = True

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        mapped = []
        for t in tools:
            # Responses 扁平定义：{type:"function", name, description, parameters}
            if isinstance(t, dict) and t.get("type") == "function" and t.get("name"):
                mapped.append({
                    "name": str(t["name"]),
                    "description": str(t.get("description") or ""),
                    "input_schema": t.get("parameters") if isinstance(t.get("parameters"), dict) else {"type": "object"},
                })
            # {"type":"function","function":{...}} 的 chat 风格也顺手兼容
            elif isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name"):
                fn = t["function"]
                mapped.append({
                    "name": str(fn["name"]),
                    "description": str(fn.get("description") or ""),
                    "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"},
                })
        if mapped:
            body["tools"] = mapped

    choice = payload.get("tool_choice")
    if choice == "none":
        body.pop("tools", None)
    elif choice == "required":
        body["tool_choice"] = {"type": "any"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = str(choice.get("name") or "")
        if name:
            body["tool_choice"] = {"type": "tool", "name": name}
    return body, None


def _response_object(resp_id: str, created: int, model: str, output: list[dict],
                     usage: dict, stop_reason: str | None) -> dict:
    status = _STOP_STATUS_MAP.get(stop_reason, "completed")
    obj: dict = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        },
    }
    if status == "incomplete":
        obj["incomplete_details"] = {"reason": "max_output_tokens"}
    return obj


def anthropic_to_responses(data: dict, model: str) -> dict:
    """Anthropic message 响应 → Responses 响应对象。"""
    output: list[dict] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": block["text"], "annotations": []}],
            })
        elif block.get("type") == "tool_use":
            bid = str(block.get("id") or "")
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": bid,
                "name": str(block.get("name") or ""),
                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                "status": "completed",
            })
    usage = data.get("usage") or {}
    return _response_object(
        str(data.get("id") or f"resp_{uuid.uuid4().hex[:24]}"), int(time.time()),
        str(data.get("model") or model), output, usage, data.get("stop_reason"),
    )


class ResponsesStreamConverter:
    """Anthropic SSE 事件流 → OpenAI Responses 事件流（有状态转换器）。

    用法：先 start() 产出 response.created，逐条 feed(event_dict) 收输出行，
    流收尾时调用 finish_fallback() 兜底补发终态事件（message_stop 已正常
    处理时返回空，流被截断/异常时补 response.incomplete/response.failed），
    保证客户端不会挂等终态。
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.seq = 0
        self.stop_reason: str | None = None
        self.usage = {"input_tokens": None, "output_tokens": None}
        self.output_items: list[dict] = []
        self._block: dict | None = None  # 当前未闭合的输出项状态
        self._started = False  # 是否收到过 message_start（有真实进度）
        self._terminal_emitted = False  # 是否已发出 completed/incomplete/failed

    def _event(self, etype: str, payload: dict) -> str:
        self.seq += 1
        body = {"type": etype, "sequence_number": self.seq, **payload}
        return f"event: {etype}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    def _response(self) -> dict:
        return _response_object(self.response_id, self.created, self.model,
                                self.output_items, self.usage, self.stop_reason)

    def _incomplete_response(self) -> dict:
        obj = self._response()
        obj["status"] = "incomplete"
        obj.pop("incomplete_details", None)  # 截断非 max_tokens，不套该 reason
        return obj

    def start(self) -> str:
        return self._event("response.created", {"response": self._response()})

    def finish_fallback(self, *, failed: bool = False) -> list[str]:
        """流收尾兜底：仅在尚未发出终态事件时补发一个，幂等。

        failed=True（异常分支）→ response.failed；正常 EOF 截断：有真实进度
        （收到过 message_start）→ response.incomplete，否则 → response.failed。
        """
        if self._terminal_emitted:
            return []
        self._terminal_emitted = True
        if failed or not self._started:
            obj = self._response()
            obj["status"] = "failed"
            return [self._event("response.failed", {"response": obj})]
        return [self._event("response.incomplete", {"response": self._incomplete_response()})]

    def feed(self, evt: dict) -> list[str]:
        etype = evt.get("type")
        if etype == "message_start":
            self._started = True
            u = (evt.get("message") or {}).get("usage") or {}
            self.usage["input_tokens"] = _as_int(u.get("input_tokens"))
            return []
        if etype == "content_block_start":
            block = evt.get("content_block") or {}
            btype = block.get("type")
            if btype == "text":
                self._block = {"kind": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "text": ""}
                item = {"type": "message", "id": self._block["id"], "role": "assistant",
                        "status": "in_progress", "content": []}
                return [
                    self._event("response.output_item.added", {"output_index": len(self.output_items), "item": item}),
                    self._event("response.content_part.added", {
                        "item_id": self._block["id"], "output_index": len(self.output_items),
                        "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
                ]
            if btype == "tool_use":
                call_id = str(block.get("id") or "")
                self._block = {"kind": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}",
                               "call_id": call_id, "name": str(block.get("name") or ""), "args": ""}
                item = {"type": "function_call", "id": self._block["id"], "call_id": call_id,
                        "name": self._block["name"], "arguments": "", "status": "in_progress"}
                return [self._event("response.output_item.added",
                                    {"output_index": len(self.output_items), "item": item})]
            return []
        if etype == "content_block_delta":
            delta = evt.get("delta") or {}
            idx = len(self.output_items)
            if self._block is None:
                return []
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                self._block["text"] += delta["text"]
                return [self._event("response.output_text.delta", {
                    "item_id": self._block["id"], "output_index": idx, "content_index": 0,
                    "delta": delta["text"]})]
            if delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                self._block["args"] += delta["partial_json"]
                return [self._event("response.function_call_arguments.delta", {
                    "item_id": self._block["id"], "output_index": idx,
                    "delta": delta["partial_json"]})]
            return []
        if etype == "content_block_stop":
            if self._block is None:
                return []
            block, idx = self._block, len(self.output_items)
            self._block = None
            if block["kind"] == "message":
                item = {"type": "message", "id": block["id"], "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": block["text"], "annotations": []}]}
                self.output_items.append(item)
                return [
                    self._event("response.output_text.done", {
                        "item_id": block["id"], "output_index": idx, "content_index": 0,
                        "text": block["text"]}),
                    self._event("response.content_part.done", {
                        "item_id": block["id"], "output_index": idx, "content_index": 0,
                        "part": {"type": "output_text", "text": block["text"], "annotations": []}}),
                    self._event("response.output_item.done", {"output_index": idx, "item": item}),
                ]
            item = {"type": "function_call", "id": block["id"], "call_id": block["call_id"],
                    "name": block["name"], "arguments": block["args"], "status": "completed"}
            self.output_items.append(item)
            return [
                self._event("response.function_call_arguments.done", {
                    "item_id": block["id"], "output_index": idx, "arguments": block["args"]}),
                self._event("response.output_item.done", {"output_index": idx, "item": item}),
            ]
        if etype == "message_delta":
            delta = evt.get("delta") or {}
            self.stop_reason = delta.get("stop_reason")
            out = _as_int((evt.get("usage") or {}).get("output_tokens"))
            if out is not None:
                self.usage["output_tokens"] = out
            return []
        if etype == "message_stop":
            # message_stop 前 content_block_stop 已把所有项收进 output_items
            self._terminal_emitted = True
            return [self._event("response.completed", {"response": self._response()})]
        if etype == "error":
            self._terminal_emitted = True
            err = evt.get("error") or {}
            return [self._event("response.failed", {"response": self._response(),
                                                    "error": err if isinstance(err, dict) else {"message": str(err)}})]
        return []  # ping / content_block_start(空) / message_start(已处理) 等
