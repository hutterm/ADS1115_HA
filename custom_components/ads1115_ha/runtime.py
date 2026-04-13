"""Runtime helpers for ADS1115 config entries."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_INTERVAL, DEFAULT_INTERVAL, DOMAIN


def _entry_interval(entry: ConfigEntry) -> int:
    return max(
        1,
        int(entry.options.get(CONF_INTERVAL, entry.data.get(CONF_INTERVAL, DEFAULT_INTERVAL))),
    )


def ensure_entry_runtime(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Create or refresh runtime data for one config entry."""
    domain_data: dict[str, dict[str, Any]] = hass.data.setdefault(DOMAIN, {})
    runtime = domain_data.setdefault(entry.entry_id, {})
    runtime[CONF_INTERVAL] = _entry_interval(entry)
    return runtime


def get_entry_runtime(hass: HomeAssistant, entry_id: str) -> dict[str, Any] | None:
    """Return runtime data for a config entry, if available."""
    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, dict):
        return None

    runtime = domain_data.get(entry_id)
    return runtime if isinstance(runtime, dict) else None


def get_runtime_interval(runtime: dict[str, Any] | None, fallback: int = DEFAULT_INTERVAL) -> int:
    """Return normalized runtime update interval in seconds."""
    if runtime is None:
        return max(1, int(fallback))
    return max(1, int(runtime.get(CONF_INTERVAL, fallback)))


def set_runtime_interval(runtime: dict[str, Any], interval_s: int) -> None:
    """Set runtime update interval in seconds."""
    runtime[CONF_INTERVAL] = max(1, int(interval_s))
