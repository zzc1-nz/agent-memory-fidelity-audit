# local/ - not published

This directory holds machine-local helpers that must **not** be committed to a public
repository, and it is listed in `.gitignore`.

## `refresh_trae.py`

Acquires a short-lived access token for the private gateway that
`memfaith/trae_client.py` speaks to. It reads an existing, already-authorized local
login for that service from the desktop application's own encrypted credential store
and exchanges the refresh token through the provider's official token endpoint.

**Why it is not published:** it contains the provider's storage-decryption parameters
and reproduces a first-party OAuth exchange. Publishing it would be pointless for a
reader (it only works on a machine with that application signed in) and it would
document a bypass-shaped shortcut around the provider's normal sign-in flow. The
benchmark does not depend on it — everything in `memfaith/` runs against any
OpenAI-compatible endpoint.

**If you want to reproduce the reported run on this backend:** obtain a credential
through the provider's supported sign-in flow and write it to
`.cache/trae_credential.json` with this shape:

```json
{
  "token": "<access token>",
  "tokenExpireAt": 1795531649396,
  "refreshToken": "<optional>",
  "region": "cn",
  "deviceId": "<optional, used in identity headers>",
  "machineId": "<optional, used in identity headers>"
}
```

Then `python -m memfaith.run ...` will pick the adapter automatically when
`MEMFAITH_BASE_URL` / `MEMFAITH_API_KEY` are unset.

**The recommended path for readers and for anyone re-running this benchmark** is an
OpenAI-compatible endpoint:

```bash
export MEMFAITH_BASE_URL=https://api.openai.com/v1
export MEMFAITH_API_KEY=sk-...
export MEMFAITH_MODEL_SUMMARIZER=gpt-4o-mini
export MEMFAITH_MODEL_ANSWERER=gpt-4o-mini
export MEMFAITH_MODEL_JUDGE=gpt-4o        # must differ from the answerer
```
