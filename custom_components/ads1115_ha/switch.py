"""Switch entities for ADS1115 integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_I2C_BUS,
    CONF_READOUT_ENABLED,
    DEFAULT_I2C_ADDRESS,
    DEFAULT_I2C_BUS,
    DEFAULT_NAME,
    DOMAIN,
)
from .runtime import (
    ensure_entry_runtime,
    get_runtime_readout_enabled,
    set_runtime_readout_enabled,
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ADS1115 switch entities from a config entry."""
    runtime_data = ensure_entry_runtime(hass, entry)
    async_add_entities([ADS1115ReadoutEnabledSwitch(entry, runtime_data)])


class ADS1115ReadoutEnabledSwitch(SwitchEntity):
    """Switch entity that controls ADS1115 channel readout."""

    _attr_has_entity_name = True
    _attr_name = "Readout"
    _attr_icon = "mdi:chart-line"

    def __init__(self, entry: ConfigEntry, runtime_data: dict[str, Any]) -> None:
        self._entry = entry
        self._runtime_data = runtime_data
        self._attr_unique_id = f"ads1115_{entry.entry_id}_readout"

        bus = int(entry.data.get(CONF_I2C_BUS, DEFAULT_I2C_BUS))
        address = int(entry.data.get(CONF_ADDRESS, DEFAULT_I2C_ADDRESS))
        device_name = str(entry.data.get(CONF_NAME, DEFAULT_NAME))
        self._attr_device_info = {
            "identifiers": {(DOMAIN, f"{bus}:{address}")},
            "name": f"{device_name} ({bus}:0x{address:02X})",
            "manufacturer": "Texas Instruments",
            "model": "ADS1115",
        }

    @property
    def is_on(self) -> bool:
        """Return true when readout is enabled."""
        return bool(get_runtime_readout_enabled(self._runtime_data))

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Enable readout."""
        set_runtime_readout_enabled(self._runtime_data, True)
        options = dict(self._entry.options)
        options[CONF_READOUT_ENABLED] = True
        self.hass.config_entries.async_update_entry(self._entry, options=options)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disable readout."""
        set_runtime_readout_enabled(self._runtime_data, False)
        options = dict(self._entry.options)
        options[CONF_READOUT_ENABLED] = False
        self.hass.config_entries.async_update_entry(self._entry, options=options)
        self.async_write_ha_state()
