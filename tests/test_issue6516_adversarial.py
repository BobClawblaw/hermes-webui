"""Adversarial regression tests for #6516: slug collision, x-ai identity.

Covers three follow-up defects found by Codex re-gate:

  1. Lossy slugging collision: two ``custom_providers`` names that slugify
     to the same string (e.g. "foo:bar" and "foo-bar") must not silently
     route to the wrong endpoint. ``_get_provider_base_url`` must return
     None (fail closed) when a slug match is ambiguous.

  2. Provider-key cleanup for colon-named entries: ``_clean_provider_key_from_config``
     must correctly match a custom provider whose ``model.provider`` is
     ``custom:192.168.5.242:8000`` (colon form) against the custom_providers
     entry whose ``name`` is ``192.168.5.242:8000``.

  3. Active-provider identity for ``x-ai``: ``_resolve_configured_provider_id``
     must preserve ``x-ai`` verbatim (it is in ``_PROVIDER_DISPLAY``) rather
     than alias-resolving it to ``xai``, so the provider card and active
     badge stay matched.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

import api.config as config
from api.providers import get_providers


@pytest.fixture(autouse=True)
def _isolate_models_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_models_cache_path", tmp_path / "models_cache.json")
    config.invalidate_models_cache()
    yield
    config.invalidate_models_cache()


# ── Issue 1: slug collision disambiguation ──────────────────────────────


def test_get_provider_base_url_unique_slug_succeeds():
    """A slug that matches exactly one custom_providers entry must return
    that entry's base_url."""
    cfg = {
        "model": {"default": "default-model", "provider": "custom:node-a-8000"},
        "custom_providers": [
            {"name": "node-a:8000", "base_url": "http://node-a:8000/v1"},
        ],
    }
    old_cfg = dict(config.cfg)
    config.cfg.clear()
    config.cfg.update(json.loads(json.dumps(cfg)))
    try:
        url = config._get_provider_base_url("custom:node-a-8000")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)
    assert url == "http://node-a:8000/v1", f"Expected unique slug URL, got {url!r}"


def test_get_provider_base_url_slug_collision_returns_none():
    """When two custom_providers names slugify to the same string,
    ``_get_provider_base_url`` must return None (fail closed) rather
    than silently returning the first match."""
    # "foo:bar" and "foo-bar" both slugify to "custom:foo-bar"
    cfg = {
        "model": {"default": "default-model", "provider": "custom:foo-bar"},
        "custom_providers": [
            {"name": "foo:bar", "base_url": "http://wrong:8000/v1"},
            {"name": "foo-bar", "base_url": "http://right:8000/v1"},
        ],
    }
    old_cfg = dict(config.cfg)
    config.cfg.clear()
    config.cfg.update(json.loads(json.dumps(cfg)))
    try:
        url = config._get_provider_base_url("custom:foo-bar")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)

    assert url is None, (
        f"Expected None on ambiguous slug collision, got {url!r}"
    )


def test_resolve_model_provider_default_still_works_after_collision():
    """When the active provider matches the model section, its URL must
    still resolve correctly even when a slug collision exists in the
    custom_providers list.  The active model path (#1) runs before
    the list scan (#3) so collisions don't break the default."""
    cfg = {
        "model": {"default": "default-model", "provider": "custom:my-server",
                  "base_url": "http://my-server:8000/v1"},
        "custom_providers": [
            {"name": "my:server", "base_url": "http://collision:5000/v1"},
            {"name": "my-server", "base_url": "http://my-server:8000/v1"},
        ],
    }
    old_cfg = dict(config.cfg)
    config.cfg.clear()
    config.cfg.update(json.loads(json.dumps(cfg)))
    try:
        # This should return the model section URL, not the first
        # custom_providers entry's URL.
        url = config._get_provider_base_url("custom:my-server")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)

    # The model section match (#2) runs before the list scan (#3),
    # so even though the list has an entry for "custom:my-server",
    # the model block is authoritative for the active provider.
    assert url == "http://my-server:8000/v1", (
        f"Expected the active model's base_url, got {url!r}"
    )


# ── Issue 2: provider-key cleanup with colon-named entries ──────────────


def test_custom_provider_name_matches_colon_against_hyphen():
    """``_custom_provider_name_matches`` must return True when
    provider_id uses colons (e.g. ``custom:192.168.5.242:8000``)
    and the entry name uses a colon that slugifies to a hyphen."""
    from api.providers import _custom_provider_name_matches

    # provider_id = "custom:192.168.5.242:8000" (with colon)
    # name = "192.168.5.242:8000" → slug = "custom:192.168.5.242-8000"
    assert _custom_provider_name_matches(
        "custom:192.168.5.242:8000", "192.168.5.242:8000"
    ), "Colon-bearing provider_id should match colon-bearing entry name"


def test_custom_provider_name_matches_hyphen_against_colon():
    """``_custom_provider_name_matches`` must return True when
    provider_id uses hyphens (e.g. ``custom:192.168.5.242-8000``)
    and the entry name uses colons."""
    from api.providers import _custom_provider_name_matches

    assert _custom_provider_name_matches(
        "custom:192.168.5.242-8000", "192.168.5.242:8000"
    ), "Hyphenated provider_id should match colon-bearing entry name"


def test_custom_provider_name_matches_no_match():
    """``_custom_provider_name_matches`` must return False for an
    unrelated provider_id."""
    from api.providers import _custom_provider_name_matches

    assert not _custom_provider_name_matches(
        "openai", "192.168.5.242:8000"
    ), "openai should not match a custom provider name"


# ── Issue 3: x-ai provider identity preserved ───────────────────────────


def test_resolve_configured_provider_id_preserves_x_ai():
    """``_resolve_configured_provider_id`` must return ``x-ai`` verbatim
    since it is in ``_PROVIDER_DISPLAY``, not alias-resolve it to ``xai``."""
    result = config._resolve_configured_provider_id("x-ai")
    assert result == "x-ai", (
        f"Expected 'x-ai' preserved verbatim, got {result!r}"
    )


def test_resolve_configured_provider_id_preserves_known_canonicals():
    """Other known canonical IDs must also be preserved verbatim."""
    for pid in ("anthropic", "openai", "deepseek", "google", "openai-codex"):
        result = config._resolve_configured_provider_id(pid)
        assert result == pid, (
            f"Expected {pid!r} preserved, got {result!r}"
        )


def test_resolve_configured_provider_id_still_aliases_unknown():
    """An unknown ID that does NOT exist in _PROVIDER_DISPLAY or
    _PROVIDER_MODELS must still be alias-resolved."""
    result = config._resolve_configured_provider_id("google-gemini")
    # google-gemini is not in _PROVIDER_DISPLAY/_PROVIDER_MODELS directly
    # (google is).  It should be alias-resolved.
    assert result != "google-gemini", (
        f"Expected alias resolution for 'google-gemini', got {result!r}"
    )


def test_get_providers_active_provider_x_ai(tmp_path, monkeypatch):
    """``get_providers()`` must report ``x-ai`` as the active provider
    when ``model.provider`` is ``x-ai``, not alias-resolved ``xai``."""
    import api.config as cfg_mod

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  default: grok-4.20\n"
        "  provider: x-ai\n"
        "  base_url: https://api.x.ai/v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg_mod, "_get_config_path", lambda: config_path)
    cfg_mod.reload_config()

    try:
        result = get_providers()
    finally:
        cfg_mod.reload_config()

    ap = result.get("active_provider")
    assert ap == "x-ai", (
        f"Expected active_provider 'x-ai', got {ap!r}. "
        "x-ai must not be alias-resolved to xai."
    )


# ── Issue 4: two-profile production-composition regression ──────────────────


def test_resolve_custom_provider_runtime_overrides_uses_config_data():
    """``_resolve_custom_provider_runtime_overrides`` must consult the
    ``config_data`` dict (the target profile's config) rather than the
    ambient ``get_config()`` when resolving a named ``custom:*`` provider.

    This is the production-composition regression: when the streaming worker
    runs under profile A but a session is routed to profile B, the initial
    send and both credential self-heal retries must pick up profile B's
    URL/key sentinels, not profile A's.
    """
    from api.streaming import _resolve_custom_provider_runtime_overrides

    profile_a_cfg = {
        "model": {"default": "model-a", "provider": "custom:worker",
                  "base_url": "http://profile-a:8000/v1"},
        "custom_providers": [
            {"name": "worker", "base_url": "http://profile-a:8000/v1",
             "api_key": "profile-a-key"},
        ],
    }
    profile_b_cfg = {
        "model": {"default": "model-b", "provider": "custom:worker",
                  "base_url": "http://profile-b:9000/v1"},
        "custom_providers": [
            {"name": "worker", "base_url": "http://profile-b:9000/v1",
             "api_key": "profile-b-key"},
        ],
    }

    # Initial send: must use profile B's config, not the ambient (A) config.
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom:worker", None, None,
        config_data=profile_b_cfg,
    )
    assert url == "http://profile-b:9000/v1", (
        f"Initial send: expected profile-b URL, got {url!r}"
    )
    assert key == "profile-b-key", (
        f"Initial send: expected profile-b key, got {key!r}"
    )
    assert provider == "custom", (
        f"Initial send: expected collapsed 'custom' provider, got {provider!r}"
    )

    # Retry path 1 (self-heal on 401): same config_data must still win.
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom:worker", None, None,
        config_data=profile_b_cfg,
    )
    assert url == "http://profile-b:9000/v1", (
        f"Retry 1: expected profile-b URL, got {url!r}"
    )
    assert key == "profile-b-key", (
        f"Retry 1: expected profile-b key, got {key!r}"
    )

    # Retry path 2 (except-path self-heal): same config_data must still win.
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom:worker", None, None,
        config_data=profile_b_cfg,
    )
    assert url == "http://profile-b:9000/v1", (
        f"Retry 2: expected profile-b URL, got {url!r}"
    )
    assert key == "profile-b-key", (
        f"Retry 2: expected profile-b key, got {key!r}"
    )


# ── Issue 6: URL-less collision fail-closed ───────────────────────────────────


def test_get_provider_base_url_url_less_collision_fails_closed():
    """When two custom_providers entries slugify to the same slug but one has
    a blank base_url, _get_provider_base_url must return None — NOT the
    sibling's URL.

    Before the fix, slug_matches only appended URL-bearing entries, so the
    URL-less colliding entry was invisible to the len > 1 check and the
    sibling's URL was returned.  After the fix, ALL slug-matching entries
    are counted, so the collision is detected and None is returned.
    """
    import api.config as config
    import json

    cfg = {
        "model": {"default": "test-model", "provider": "openai"},
        "custom_providers": [
            {"name": "foo:bar", "base_url": "http://sibling.example/v1",
             "api_key": "sibling-key"},
            {"name": "foo-bar", "base_url": "", "api_key": ""},
        ],
    }
    old_cfg = dict(config.cfg)
    config.cfg.clear()
    config.cfg.update(json.loads(json.dumps(cfg)))
    try:
        url = config._get_provider_base_url("custom:foo-bar")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)
    assert url is None, (
        f"URL-less collision must fail closed (return None), got {url!r}"
    )


def test_get_provider_base_url_single_match_with_url_returns_url():
    """When exactly one entry matches (even if a sibling with a blank URL
    exists but doesn't slug-match), the URL is returned.
    """
    import api.config as config
    import json

    cfg = {
        "model": {"default": "test-model", "provider": "openai"},
        "custom_providers": [
            {"name": "foo:bar", "base_url": "http://valid.example/v1",
             "api_key": "valid-key"},
            {"name": "other:provider", "base_url": "", "api_key": ""},
        ],
    }
    old_cfg = dict(config.cfg)
    config.cfg.clear()
    config.cfg.update(json.loads(json.dumps(cfg)))
    try:
        url = config._get_provider_base_url("custom:foo-bar")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)
    assert url == "http://valid.example/v1", (
        f"Single match should return URL, got {url!r}"
    )


# ── Issue 7: custom:slug identity preserved through retry ─────────────────────


def test_resolve_custom_provider_runtime_overrides_preserves_identity():
    """_resolve_custom_provider_runtime_overrides must collapse custom:slug
    to plain 'custom' on the initial resolution, but the caller (streaming
    worker) must retain the original custom:slug for retry paths.

    This test verifies that calling _resolve_custom_provider_runtime_overrides
    with a custom:slug provider returns 'custom' as the provider (collapsed),
    and that a second call with the collapsed 'custom' value does NOT re-enter
    the custom: resolution path (returns 'custom' unchanged).
    """
    from api.streaming import _resolve_custom_provider_runtime_overrides

    profile_cfg = {
        "model": {"default": "model-a", "provider": "custom:worker",
                  "base_url": "http://profile-a:8000/v1"},
        "custom_providers": [
            {"name": "worker", "base_url": "http://profile-a:8000/v1",
             "api_key": "profile-a-key"},
        ],
    }

    # Initial resolution: custom:worker → custom (collapsed)
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom:worker", None, None,
        config_data=profile_cfg,
    )
    assert provider == "custom", (
        f"Initial: expected collapsed 'custom', got {provider!r}"
    )
    assert url == "http://profile-a:8000/v1", (
        f"Initial: expected profile-a URL, got {url!r}"
    )
    assert key == "profile-a-key", (
        f"Initial: expected profile-a key, got {key!r}"
    )

    # Retry with collapsed 'custom': must NOT re-enter custom: path
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom", None, None,
        config_data=profile_cfg,
    )
    assert provider == "custom", (
        f"Retry: collapsed 'custom' must pass through, got {provider!r}"
    )
    assert url is None, (
        f"Retry: collapsed 'custom' must not resolve URL, got {url!r}"
    )

    # Retry with original custom:worker: must re-resolve from config_data
    provider, key, url = _resolve_custom_provider_runtime_overrides(
        "custom:worker", None, None,
        config_data=profile_cfg,
    )
    assert provider == "custom", (
        f"Retry: expected collapsed 'custom', got {provider!r}"
    )
    assert url == "http://profile-a:8000/v1", (
        f"Retry: expected profile-a URL, got {url!r}"
    )
    assert key == "profile-a-key", (
        f"Retry: expected profile-a key, got {key!r}"
    )
