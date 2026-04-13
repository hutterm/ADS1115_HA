"""Number entities for ADS1115 integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.number import NumberDeviceClass, NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS, CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_I2C_BUS, CONF_INTERVAL, DEFAULT_I2C_ADDRESS, DEFAULT_I2C_BUS, DEFAULT_NAME, DOMAIN
from .runtime import ensure_entry_runtime, get_runtime_interval, set_runtime_interval


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ADS1115 number entities from a config entry."""
    runtime_data = ensure_entry_runtime(hass, entry)
    async_add_entities([ADS1115UpdateIntervalNumber(entry, runtime_data)])


class ADS1115UpdateIntervalNumber(NumberEntity):
    """Number entity that controls ADS1115 polling interval."""

    _attr_has_entity_name = True
    _attr_name = "Update interval"
    _attr_icon = "mdi:timer-cog"
    _attr_mode = NumberMode.BOX
    _attr_device_class = NumberDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_native_min_value = 1
    _attr_native_max_value = 3600
    _attr_native_step = 1

    def __init__(self, entry: ConfigEntry, runtime_data: dict[str, Any]) -> None:
        self._entry = entry
        self._runtime_data = runtime_data
        self._attr_unique_id = f"ads1115_{entry.entry_id}_update_interval"

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
    def native_value(self) -> float:
        """Return current update interval in seconds."""
        return float(get_runtime_interval(self._runtime_data))

    async def async_set_native_value(self, value: float) -> None:
        """Update the runtime and persisted interval."""
        interval = max(1, int(round(value)))
        set_runtime_interval(self._runtime_data, interval)

        options = dict(self._entry.options)
        options[CONF_INTERVAL] = interval
        self.hass.config_entries.async_update_entry(self._entry, options=options)

        self.async_write_ha_state()
