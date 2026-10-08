"""GW-GEM /v1beta/models/*:generateContent（Gemini 风格）端点集成测试。

复用 /v1/messages 同一套 mock 上游（返回 Anthropic 格式），验证双向转换：
非流式 JSON 与 ?alt=sse 流式 GenerateContentResponse chunk。
"""

from __future__ import annotations

import json

import pytest

_GOOD_JWT = "hO.eyJzdWIiOiJvIn0.sig"


def _parse_sse(text: str) -> list[dict]:
    return [json.loads(ln[6:]) for ln in text.splitlines() if ln.startswith("data: ")]


@pytest.mark.integration
class TestGeminiGenerateContent:
    async def test_nonstream_basic(self, gateway_client, fresh_app):
        client, mock = gateway_client
        from tests.conftest import seed_account

        seed_account(fresh_app, _GOOD_JWT, name="gem")
        res = await client.post(
            "/v1beta/models/glm-5.3-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["modelVersion"] == "GLM-5.3-Flash"
        cand = data["candidates"][0]
        assert cand["content"]["role"] == "model"
        assert cand["content"]["parts"] == [{"text": "Hello from mock upstream"}]
        assert cand["finishReason"] == "STOP"
        assert data["usageMetadata"]["promptTokenCount"] == 10

    async def test_model_name_normalized_from_path(self, gateway_client, fresh_app):
        """路径段小写别名 → 官方名（MODEL_NAME_MAP），URL 编码也容忍。"""
        client, mock = gateway_client
        from tests.conftest import seed_account

        seed_account(fresh_app, _GOOD_JWT, name="gem-alias")
        res = await client.post(
            "/v1beta/models/glm-5.3:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert res.status_code == 200
        assert res.json()["modelVersion"] == "GLM-5.3"

    async def test_system_instruction_reaches_upstream_as_system(self, gateway_client, fresh_app):
        client, mock = gateway_client
        from tests.conftest import seed_account

        seed_account(fresh_app, _GOOD_JWT, name="gem-sys")
        res = await client.post(
            "/v1beta/models/GLM-5.3:generateContent",
            json={"systemInstruction": {"parts": [{"text": "守则"}]},
                  "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert res.status_code == 200
        payload = json.loads(mock.state.calls[-1][3])
        assert payload["system"][-1]["text"] == "守则"  # 用户 system 位于官方身份块之后
        assert payload["max_tokens"] == 32768

    async def test_stream_returns_gemini_chunks(self, gateway_client, fresh_app):
        client, mock = gateway_client
        from tests.conftest import seed_account

        seed_account(fresh_app, _GOOD_JWT, name="gem-stream")
        res = await client.post(
            "/v1beta/models/glm-5.3-flash:streamGenerateContent?alt=sse",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        chunks = _parse_sse(res.text)
        assert chunks, "SSE 流不得为空（agy 空响应会无限重试）"
        # 内容 chunk：text parts 非空
        text = "".join(
            p["text"] for c in chunks for p in c["candidates"][0]["content"]["parts"] if p.get("text")
        )
        assert "chunk-0" in text
        # 终态 chunk：finishReason + usageMetadata
        finals = [c for c in chunks if c["candidates"][0].get("finishReason")]
        assert len(finals) == 1
        assert finals[0]["candidates"][0]["finishReason"] == "STOP"
        assert finals[0]["usageMetadata"]["candidatesTokenCount"] == 5
        assert finals[0]["usageMetadata"]["totalTokenCount"] == 5  # mock 未上报 prompt 侧
        # 每个 chunk 的 parts 非空（agy 空输出防护）
        assert all(c["candidates"][0]["content"]["parts"] for c in chunks)
        # 上游收到 stream=true（alt=sse 透传）
        payload = json.loads(mock.state.calls[-1][3])
        assert payload["stream"] is True

    async def test_invalid_payload_returns_gemini_error(self, gateway_client, fresh_app):
        client, _ = gateway_client
        res = await client.post("/v1beta/models/glm-5.3:generateContent",
                                content=b"not json", headers={"content-type": "application/json"})
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == 400 and err["status"] == "INVALID_ARGUMENT"

    async def test_no_account_returns_503_gemini_error(self, fresh_app):
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/v1beta/models/glm-5.3:generateContent",
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )
        assert res.status_code == 503
        assert res.json()["error"]["status"] == "UNAVAILABLE"

    async def test_gateway_key_required(self, gateway_client, fresh_app):
        client, _ = gateway_client
        fresh_app.set_setting("gateway_key", "sk-gw-test")
        try:
            res = await client.post(
                "/v1beta/models/glm-5.3:streamGenerateContent?alt=sse",
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )
            assert res.status_code == 401
            # agy 的 Bearer 头形态直接复用
            res = await client.post(
                "/v1beta/models/glm-5.3:streamGenerateContent?alt=sse",
                headers={"Authorization": "Bearer sk-gw-test"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )
            assert res.status_code == 503  # 鉴权通过，进入调度（无账号）
        finally:
            fresh_app.set_setting("gateway_key", "")
