"""captcha 逃生门（ZCODE_CAPTCHA_PREFIX/REGION/SCENE_ID）解析口径。

守护 805fcf6 遗留的三件事：
- 单一真源 resolve_solver_params：env > 远端 client/configs > CAPTCHA_DEFAULTS
- 空白值不得进 solver argv（strip + 空值回退，对齐 settings 既有约定）
- solver 实参、_Token.region（→ 上报上游 region 头）、/claim/captcha-config
  三者必须一致，手动领取路径不能绕过覆盖
"""

from __future__ import annotations

import pytest

from app import captcha as captcha_module
from app import constants
from app.captcha import CaptchaManager, resolve_solver_params

REMOTE_CONFIG = {"enabled": True, "prefix": "no8xfe", "region": "sgp", "sceneId": "remote-scene"}


@pytest.fixture(autouse=True)
def _reset_override_log_state(monkeypatch):
    """env 覆盖告警有跨调用去重态，逐用例复位避免相互影响。"""
    monkeypatch.setattr(captcha_module, "_logged_override", None)


def test_no_override_falls_back_to_remote_config():
    scene, region, prefix = resolve_solver_params(REMOTE_CONFIG)
    assert (scene, region, prefix) == ("remote-scene", "sgp", "no8xfe")


def test_env_override_wins_over_remote_config(monkeypatch):
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    monkeypatch.setenv("ZCODE_CAPTCHA_REGION", "cn")
    monkeypatch.setenv("ZCODE_CAPTCHA_SCENE_ID", "11xygtvd")
    assert resolve_solver_params(REMOTE_CONFIG) == ("11xygtvd", "cn", "8ab4")


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
@pytest.mark.parametrize("env_name", ["ZCODE_CAPTCHA_PREFIX", "ZCODE_CAPTCHA_REGION",
                                      "ZCODE_CAPTCHA_SCENE_ID"])
def test_blank_env_falls_back_not_passed_through(monkeypatch, env_name, blank):
    """空串/纯空白 = 未覆盖：既不能盖掉远端值，也不能带着空白进 argv。"""
    monkeypatch.setenv(env_name, blank)
    assert resolve_solver_params(REMOTE_CONFIG) == ("remote-scene", "sgp", "no8xfe")


def test_override_value_is_stripped(monkeypatch):
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "  8ab4  ")
    monkeypatch.setenv("ZCODE_CAPTCHA_REGION", " cn ")
    scene, region, prefix = resolve_solver_params(REMOTE_CONFIG)
    assert (region, prefix) == ("cn", "8ab4")
    assert scene == "remote-scene"  # 未设 scene 时保留远端值


def test_empty_config_uses_defaults():
    scene, region, prefix = resolve_solver_params({})
    defaults = constants.CAPTCHA_DEFAULTS
    assert (scene, region, prefix) == (defaults["sceneId"], defaults["region"], defaults["prefix"])


def test_partial_override_only_prefix(monkeypatch):
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    scene, region, prefix = resolve_solver_params(REMOTE_CONFIG)
    assert (scene, region, prefix) == ("remote-scene", "sgp", "8ab4")


class _Recorder:
    def __init__(self, param: str | None = "verify-param") -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.param = param

    async def __call__(self, scene: str, region: str, prefix: str) -> str | None:
        self.calls.append((scene, region, prefix))
        return self.param


async def test_solve_one_feeds_solver_and_token_same_region(monkeypatch):
    """逃生门生效时：solver argv 与 _Token.region 必须是同一个值（防上报头分叉）。"""
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    monkeypatch.setenv("ZCODE_CAPTCHA_REGION", "cn")

    manager = CaptchaManager()
    recorder = _Recorder()
    monkeypatch.setattr(manager, "_run_solver", recorder)
    token = await manager._solve_one(REMOTE_CONFIG)

    assert token is not None
    assert recorder.calls == [("remote-scene", "cn", "8ab4")]
    assert (token.param, token.region) == ("verify-param", "cn")


async def test_solve_one_success_log_reports_effective_params(monkeypatch, capsys):
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    monkeypatch.setenv("ZCODE_CAPTCHA_REGION", "cn")

    manager = CaptchaManager()
    monkeypatch.setattr(manager, "_run_solver", _Recorder())
    await manager._solve_one(REMOTE_CONFIG)

    out = capsys.readouterr().out
    assert "prefix/region/sceneId 被环境变量强制覆盖" in out
    assert "prefix=8ab4" in out and "region=cn" in out
    assert "scene=remote-scene region=cn prefix=8ab4" in out


async def test_no_override_no_warn(monkeypatch, capsys):
    manager = CaptchaManager()
    monkeypatch.setattr(manager, "_run_solver", _Recorder())
    await manager._solve_one(REMOTE_CONFIG)
    assert "被环境变量强制覆盖" not in capsys.readouterr().out


async def test_override_warn_logged_once_until_value_changes(monkeypatch, capsys):
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    manager = CaptchaManager()
    monkeypatch.setattr(manager, "_run_solver", _Recorder())

    await manager._solve_one(REMOTE_CONFIG)
    await manager._solve_one(REMOTE_CONFIG)
    assert capsys.readouterr().out.count("被环境变量强制覆盖") == 1

    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "beef")
    await manager._solve_one(REMOTE_CONFIG)
    second = capsys.readouterr().out
    assert second.count("被环境变量强制覆盖") == 1  # 覆盖值变了 → 重新提示一次
    assert "prefix=beef" in second


async def test_manual_claim_config_endpoint_applies_override(monkeypatch):
    """pool 断供时手动领取是唯一兜底：该路径不得绕过逃生门。"""
    from app.captcha import captcha_manager
    from app.routes.admin_api import claim_captcha_config

    async def _fake_config() -> dict:
        return dict(REMOTE_CONFIG)

    monkeypatch.setattr(captcha_manager, "fetch_config", _fake_config)
    monkeypatch.setenv("ZCODE_CAPTCHA_PREFIX", "8ab4")
    monkeypatch.setenv("ZCODE_CAPTCHA_REGION", "cn")

    payload = await claim_captcha_config()
    assert payload["prefix"] == "8ab4"
    assert payload["region"] == "cn"
    assert payload["scene_id"] == "remote-scene"
    assert payload["enabled"] is True
