"""Tests for the Trae gateway adapter and the report's ASCII output.

These cover the pieces that are easy to break and hard to notice: SSE parsing of
the gateway's event stream, the credential-state guard, and the guarantee that the
generated report stays plain ASCII.
"""

from __future__ import annotations

import json

import pytest

from memfaith import trae_client


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
def test_parse_sse_concatenates_output_frames():
    raw = (
        'event:metadata\ndata:{"session_id":"abc","model":""}\n\n'
        'event:output\ndata:{"response":"Hello","reasoning_content":"think "}\n\n'
        'event:output\ndata:{"response":" world","reasoning_content":"more"}\n\n'
        "event:done\ndata:{}\n\n"
    )
    parsed = trae_client.parse_sse(raw)
    assert parsed["text"] == "Hello world"
    assert parsed["reasoning"] == "think more"
    assert parsed["errors"] == []
    assert parsed["meta"]["session_id"] == "abc"


def test_parse_sse_captures_error_frames():
    raw = (
        'event:error\ndata:{"code":4001,"message":"param is invalid"}\n\n'
    )
    parsed = trae_client.parse_sse(raw)
    assert parsed["text"] == ""
    assert len(parsed["errors"]) == 1
    assert "4001" in parsed["errors"][0]


def test_parse_sse_tolerates_malformed_and_empty_frames():
    raw = (
        "event:output\ndata:not-json\n\n"
        "event:output\ndata:{\"response\":\"ok\"}\n\n"
        "\n\n"
        "event:output\n"
    )
    parsed = trae_client.parse_sse(raw)
    assert parsed["text"] == "ok"


def test_parse_sse_empty_stream():
    parsed = trae_client.parse_sse("")
    assert parsed["text"] == "" and parsed["errors"] == []


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------
def test_load_credential_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(trae_client, "TRAE_STATE_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(trae_client.TraeAuthError) as exc:
        trae_client.load_credential()
    assert "supply your own" in str(exc.value)


def test_load_credential_expired_token_raises(tmp_path, monkeypatch):
    path = tmp_path / "cred.json"
    path.write_text(json.dumps({"token": "abc", "tokenExpireAt": 1}), encoding="utf-8")
    monkeypatch.setattr(trae_client, "TRAE_STATE_PATH", str(path))
    with pytest.raises(trae_client.TraeAuthError) as exc:
        trae_client.load_credential()
    assert "expired" in str(exc.value)


def test_load_credential_valid(tmp_path, monkeypatch):
    path = tmp_path / "cred.json"
    path.write_text(json.dumps({
        "token": "abc", "tokenExpireAt": 9_999_999_999_999,
        "deviceId": "d", "machineId": "m", "region": "cn",
    }), encoding="utf-8")
    monkeypatch.setattr(trae_client, "TRAE_STATE_PATH", str(path))
    cred = trae_client.load_credential()
    assert cred["token"] == "abc"


# ---------------------------------------------------------------------------
# Role separation (the self-bias control)
# ---------------------------------------------------------------------------
def test_role_models_are_distinct_where_it_matters():
    """The judge must never be the same model as the answerer."""
    models = trae_client.ROLE_MODELS
    assert models["judge"] != models["answerer"], (
        "judge and answerer share a model id; the audit would grade its own output")
    assert models["summarizer"] != models["answerer"]
    assert set(models) >= {"summarizer", "answerer", "judge", "extractive"}


def test_sse_body_shape_matches_gateway_contract():
    body = trae_client.TraeClient._body(
        [{"role": "user", "content": "hi"}], "m", "solo_work_lite", 100, 0.0)
    assert body["model"] == "m"
    assert body["config_name"] == "m"       # the wire id the gateway accepts
    assert body["function"] == "solo_work_lite"
    assert body["stream"] is True
    # content must be a list of typed parts, not a bare string
    assert body["messages"][0]["content"][0]["type"] == "text"


# ---------------------------------------------------------------------------
# Report ASCII guarantee
# ---------------------------------------------------------------------------
def test_report_ascii_fallback_translates_punctuation():
    from memfaith.report.build import _ASCII_FALLBACK

    text = "a \u2014 b \u2013 c \u201cquoted\u201d \u2026 x \u00d7 y"
    out = text.translate(_ASCII_FALLBACK)
    assert all(ord(ch) < 128 for ch in out)
    assert "a - b" in out and '"quoted"' in out
