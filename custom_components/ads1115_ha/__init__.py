"""ADS1115 ADC Sensor integration for Home Assistant."""

from collections.abc import Mapping
import logging

from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN, LEGACY_YAML_DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR]


async def async_setup(hass: HomeAssistant, config) -> bool:
    """Set up the ADS1115 component."""
    legacy_config = config.get(LEGACY_YAML_DOMAIN)
    if legacy_config is None:
        return True

    legacy_entries = legacy_config if isinstance(legacy_config, list) else [legacy_config]
    for entry in legacy_entries:
        if not isinstance(entry, Mapping):
            _LOGGER.warning("Ignoring invalid %s YAML entry: %s", LEGACY_YAML_DOMAIN, entry)
            continue
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": SOURCE_IMPORT},
                data=dict(entry),
            )
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up ADS1115 from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload ADS1115 config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
