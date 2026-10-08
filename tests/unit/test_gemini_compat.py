"""gemini_compat 单元测试 —— Gemini generateContent ↔ Anthropic 双向转换。"""

from __future__ import annotations

import json

from app.gemini_compat import (
    GeminiStreamConverter,
    anthropic_to_gemini,
    gemini_error,
    gemini_to_anthropic,
)


class TestGeminiToAnthropic:
    def test_basic_text_and_system_instruction(self):
        body, err = gemini_to_anthropic({
            "systemInstruction": {"parts": [{"text": "你是助手"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "你好"}]},
                {"role": "model", "parts": [{"text": "你好！"}]},
                {"role": "user", "parts": [{"text": "继续"}]},
            ],
        }, "glm-5.3-flash")
        assert err is None
        assert body["model"] == "glm-5.3-flash"
        assert body["system"] == "你是助手"
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
        assert body["messages"][0]["content"] == [{"type": "text", "text": "你好"}]
        assert body["max_tokens"] == 32768  # agy 不带 maxOutputTokens 时的默认值

    def test_system_instruction_string_form(self):
        body, _ = gemini_to_anthropic({
            "systemInstruction": "规则",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }, "glm-5.3")
        assert body["system"] == "规则"

    def test_generation_config_mapping(self):
        body, _ = gemini_to_anthropic({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 128, "temperature": 0.5, "topP": 0.9,
                                 "stopSequences": ["END"], "thinkingConfig": {"includeThoughts": True}},
        }, "glm-5.3")
        assert body["max_tokens"] == 128
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9
        assert body["stop_sequences"] == ["END"]
        assert "thinkingConfig" not in json.dumps(body)  # thinkingConfig 忽略

    def test_max_output_tokens_over_limit_pre_clamped_by_normalize(self):
        """超限 maxOutputTokens 先原样带入，_normalize_body 钳制是网关层职责。"""
        body, _ = gemini_to_anthropic({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 999999},
        }, "glm-5.3")
        assert body["max_tokens"] == 999999

    def test_function_call_and_response_roundtrip_shape(self):
        body, err = gemini_to_anthropic({
            "contents": [
                {"role": "user", "parts": [{"text": "天气如何"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "杭州"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather",
                                                                 "response": {"result": "晴 25 度"}}}]},
            ],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "查天气",
                 "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
            ]}],
        }, "glm-5.3")
        assert err is None
        asst = body["messages"][1]
        assert asst["role"] == "assistant"
        assert asst["content"][0]["type"] == "tool_use"
        assert asst["content"][0]["name"] == "get_weather"
        assert asst["content"][0]["input"] == {"city": "杭州"}
        assert asst["content"][0]["id"] == "call_get_weather"
        result_msg = body["messages"][2]
        assert result_msg["content"][0]["type"] == "tool_result"
        assert result_msg["content"][0]["tool_use_id"] == "call_get_weather"  # 与 tool_use id 严格一致配对
        assert json.loads(result_msg["content"][0]["content"]) == {"result": "晴 25 度"}
        assert body["tools"] == [{"name": "get_weather", "description": "查天气",
                                  "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]

    def test_consecutive_same_role_merged(self):
        """agent 循环里 model functionCall 与后续 model 文本可能连续出现，须合并成一条。"""
        body, _ = gemini_to_anthropic({
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "a"}]},
                {"role": "model", "parts": [{"text": "b"}]},
            ],
        }, "glm-5.3")
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    def test_inline_data_image(self):
        body, _ = gemini_to_anthropic({
            "contents": [{"role": "user", "parts": [
                {"text": "看图"},
                {"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
            ]}],
        }, "glm-5.3")
        blocks = body["messages"][0]["content"]
        assert blocks[0] == {"type": "text", "text": "看图"}
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}

    def test_tool_config_modes(self):
        base = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "tools": [{"functionDeclarations": [{"name": "f"}]}]}
        body, _ = gemini_to_anthropic({
            **base, "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        }, "glm-5.3")
        assert body["tool_choice"] == {"type": "any"}
        body, _ = gemini_to_anthropic({
            **base, "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
        }, "glm-5.3")
        assert "tools" not in body

    def test_invalid_payloads(self):
        assert gemini_to_anthropic("x", "glm-5.3")[1] is not None
        body, err = gemini_to_anthropic({}, "glm-5.3")
        assert err is None  # 空 contents 宽容兜底为一条空 user 消息
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": ""}]}]


class TestAnthropicToGemini:
    def test_text_response(self):
        out = anthropic_to_gemini({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "GLM-5.3",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }, "GLM-5.3")
        cand = out["candidates"][0]
        assert cand["content"]["role"] == "model"
        assert cand["content"]["parts"] == [{"text": "Hello"}]
        assert cand["finishReason"] == "STOP"
        assert out["usageMetadata"] == {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        assert out["modelVersion"] == "GLM-5.3"

    def test_tool_use_response(self):
        out = anthropic_to_gemini({
            "id": "msg_2", "type": "message", "model": "GLM-5.3",
            "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "杭州"}}],
            "stop_reason": "tool_use", "usage": {},
        }, "GLM-5.3")
        cand = out["candidates"][0]
        assert cand["content"]["parts"] == [{"functionCall": {"name": "get_weather", "args": {"city": "杭州"}}}]
        assert cand["finishReason"] == "STOP"

    def test_max_tokens_finish_reason(self):
        out = anthropic_to_gemini({
            "id": "m", "type": "message", "content": [{"type": "text", "text": "x"}],
            "stop_reason": "max_tokens", "usage": {},
        }, "GLM-5.3")
        assert out["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_empty_content_never_yields_empty_parts(self):
        """agy 对空 parts 无限重试——兜底占位空文本。"""
        out = anthropic_to_gemini({"id": "m", "type": "message", "content": [], "usage": {}}, "GLM-5.3")
        assert out["candidates"][0]["content"]["parts"] == [{"text": ""}]


class TestGeminiStreamConverter:
    @staticmethod
    def _feed_all(conv: GeminiStreamConverter, events: list[dict]) -> list[dict]:
        outs: list[str] = []
        for evt in events:
            outs.extend(conv.feed(evt))
        outs.extend(conv.finish_fallback())
        return [json.loads(line[6:]) for line in outs if line.startswith("data: ")]

    def test_text_stream(self):
        conv = GeminiStreamConverter("GLM-5.3-Flash")
        chunks = self._feed_all(conv, [
            {"type": "message_start", "message": {"id": "msg_x", "usage": {"input_tokens": 7}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "好"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ])
        assert [c["candidates"][0]["content"]["parts"] for c in chunks[:2]] == [[{"text": "你"}], [{"text": "好"}]]
        final = chunks[-1]["candidates"][0]
        assert final["finishReason"] == "STOP"
        assert chunks[-1]["usageMetadata"] == {"promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": 9}
        assert chunks[-1]["modelVersion"] == "GLM-5.3-Flash"
        assert len(chunks) == 3  # message_stop 不再重复发终态

    def test_thinking_stream(self):
        conv = GeminiStreamConverter("GLM-5.3")
        chunks = self._feed_all(conv, [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "推理中"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "答"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ])
        assert chunks[0]["candidates"][0]["content"]["parts"] == [{"thought": True, "text": "推理中"}]
        assert chunks[1]["candidates"][0]["content"]["parts"] == [{"text": "答"}]

    def test_tool_use_stream(self):
        conv = GeminiStreamConverter("GLM-5.3")
        chunks = self._feed_all(conv, [
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "call_1", "name": "f"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a"'}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ":1}"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
            {"type": "message_stop"},
        ])
        # functionCall 在 content_block_stop 时整体输出（Gemini 无参数分片形态）
        assert {"functionCall": {"name": "f", "args": {"a": 1}}} in chunks[0]["candidates"][0]["content"]["parts"]
        assert chunks[-1]["candidates"][0]["finishReason"] == "STOP"

    def test_mixed_thinking_text_tool(self):
        conv = GeminiStreamConverter("GLM-5.3")
        chunks = self._feed_all(conv, [
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "嗯"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "查天气"}},
            {"type": "content_block_start", "index": 2,
             "content_block": {"type": "tool_use", "id": "call_1", "name": "get_weather"}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city":"北京"}'}},
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ])
        parts = [p for c in chunks for p in c["candidates"][0]["content"]["parts"]]
        assert parts[0] == {"thought": True, "text": "嗯"}
        assert parts[1] == {"text": "查天气"}
        assert parts[2] == {"functionCall": {"name": "get_weather", "args": {"city": "北京"}}}
        assert all(parts), "任何 chunk 都不得产出空 parts"

    def test_truncated_stream_fallback_emits_final_chunk(self):
        """上游截断（无 message_delta/message_stop）时 finish_fallback 兜底终态。"""
        conv = GeminiStreamConverter("GLM-5.3")
        conv.feed({"type": "message_start", "message": {"usage": {"input_tokens": 7}}})
        outs = conv.feed({"type": "content_block_delta", "index": 0,
                          "delta": {"type": "text_delta", "text": "hi"}})
        outs += conv.finish_fallback()
        chunks = [json.loads(line[6:]) for line in outs if line.startswith("data: ")]
        assert chunks[-1]["candidates"][0]["finishReason"] == "STOP"
        assert chunks[-1]["usageMetadata"]["promptTokenCount"] == 7

    def test_finish_fallback_idempotent(self):
        conv = GeminiStreamConverter("GLM-5.3")
        conv.feed({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})
        conv.feed({"type": "message_stop"})
        assert conv.finish_fallback() == []  # 终态已发，不重复

    def test_unknown_events_ignored(self):
        conv = GeminiStreamConverter("GLM-5.3")
        assert conv.feed({"type": "ping"}) == []
        assert conv.feed({"type": "content_block_start", "content_block": {"type": "text"}}) == []
        assert conv.feed({"type": "content_block_stop", "index": 0}) == []

    def test_empty_stream_still_gets_terminal_chunk(self):
        """完全空流（连 message_start 都没有）也要有终态，agy 不挂等。"""
        conv = GeminiStreamConverter("GLM-5.3")
        outs = conv.finish_fallback()
        chunks = [json.loads(line[6:]) for line in outs if line.startswith("data: ")]
        assert len(chunks) == 1
        assert chunks[0]["candidates"][0]["finishReason"] == "STOP"


class TestGeminiError:
    def test_error_shape(self):
        assert gemini_error(400, "bad") == {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}
        assert gemini_error(503, "busy")["error"]["status"] == "UNAVAILABLE"
        assert gemini_error(418, "?")["error"]["status"] == "INTERNAL"  # 未知码兜底
