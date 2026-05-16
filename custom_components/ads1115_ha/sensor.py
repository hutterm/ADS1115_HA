"""ADS1115 ADC Sensor integration for Home Assistant."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Optional

import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.core import HomeAssistant
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import (
    CONF_CHANNELS,
    CONF_CHANNEL_NUMBER,
    CONF_CLASS,
    CONF_FILTER,
    CONF_GAIN,
    CONF_I2C_BUS,
    CONF_I2C_LOCKS_KEY,
    CONF_INTERVAL,
    CONF_MAX,
    CONF_MIN,
    CONF_SCALE,
    CONF_UNIT,
    CONF_ZERO,
    DEFAULT_FILTER,
    DEFAULT_GAIN,
    DEFAULT_I2C_ADDRESS,
    DEFAULT_I2C_BUS,
    DEFAULT_I2C_LOCKS_KEY,
    DEFAULT_INTERVAL,
    DEFAULT_MAX,
    DEFAULT_MIN,
    DEFAULT_NAME,
    DEFAULT_SCALE,
    DEFAULT_UNIT,
    DOMAIN,
    GAIN_OPTIONS,
)
from .i2c_lock import get_i2c_bus_lock
from .runtime import (
    ensure_entry_runtime,
    get_entry_runtime,
    get_runtime_interval,
    get_runtime_readout_enabled,
)

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(seconds=1)


# Try to import the ADS1x15-ADC library
try:
    import ADS1x15  # import ads1115, ads1015, analogIn

    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False


def _default_channel_config(channel_number: int) -> dict[str, Any]:
    """Return default configuration for one ADS1115 channel."""
    return {
        CONF_CHANNEL_NUMBER: channel_number,
        CONF_NAME: f"ADC{channel_number}",
        CONF_UNIT: DEFAULT_UNIT,
        CONF_MIN: DEFAULT_MIN,
        CONF_MAX: DEFAULT_MAX,
        CONF_SCALE: DEFAULT_SCALE,
        CONF_ZERO: DEFAULT_MIN,
        CONF_FILTER: DEFAULT_FILTER,
    }


def _normalize_channel_config(channel_config: dict[str, Any]) -> dict[str, Any]:
    """Fill optional channel fields with defaults and validate zero bounds."""
    channel_number = int(channel_config[CONF_CHANNEL_NUMBER])
    min_val = int(channel_config.get(CONF_MIN, DEFAULT_MIN))
    max_val = int(channel_config.get(CONF_MAX, DEFAULT_MAX))
    zero_raw = channel_config.get(CONF_ZERO, min_val)
    zero = max(min_val, min(max_val, int(zero_raw)))

    return {
        CONF_CHANNEL_NUMBER: channel_number,
        CONF_NAME: channel_config.get(CONF_NAME, f"ADC{channel_number}"),
        CONF_UNIT: channel_config.get(CONF_UNIT, DEFAULT_UNIT),
        CONF_MIN: min_val,
        CONF_MAX: max_val,
        CONF_SCALE: int(channel_config.get(CONF_SCALE, DEFAULT_SCALE)),
        CONF_ZERO: zero,
        CONF_FILTER: bool(channel_config.get(CONF_FILTER, DEFAULT_FILTER)),
        CONF_CLASS: channel_config.get(CONF_CLASS),
    }


async def async_i2c_call(hass: HomeAssistant, lock, func, *args):
    """Run one blocking I2C call under the shared bus lock."""
    async with lock:
        return await hass.async_add_executor_job(func, *args)


CHANNEL_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CHANNEL_NUMBER): vol.All(vol.Coerce(int), vol.Range(min=0, max=3)),
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_UNIT, default=DEFAULT_UNIT): cv.string,
        vol.Optional(CONF_MIN, default=DEFAULT_MIN): vol.Coerce(int),
        vol.Optional(CONF_MAX, default=DEFAULT_MAX): vol.Coerce(int),
        vol.Optional(CONF_SCALE, default=DEFAULT_SCALE): vol.Coerce(int),
        vol.Optional(CONF_ZERO): vol.Coerce(int),
        vol.Optional(CONF_FILTER, default=DEFAULT_FILTER): cv.boolean,
        vol.Optional(CONF_CLASS): cv.string,
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_I2C_BUS, default=DEFAULT_I2C_BUS): vol.Coerce(int),
        vol.Optional(CONF_ADDRESS, default=DEFAULT_I2C_ADDRESS): vol.Coerce(int),
        vol.Optional(CONF_GAIN, default=DEFAULT_GAIN): vol.All(
            vol.Coerce(float),
            vol.In(GAIN_OPTIONS),
        ),
        vol.Optional(CONF_INTERVAL, default=DEFAULT_INTERVAL): vol.All(
            vol.Coerce(int),
            vol.Range(min=1),
        ),
        vol.Optional(CONF_I2C_LOCKS_KEY, default=DEFAULT_I2C_LOCKS_KEY): cv.string,
        vol.Required(CONF_CHANNELS): vol.All(cv.ensure_list, [CHANNEL_SCHEMA]),
    }
)


async def _async_build_entities(
    hass: HomeAssistant,
    *,
    name: str,
    bus: int,
    address: int,
    gain: float,
    interval: int,
    channels_config: list[dict[str, Any]],
    i2c_locks_key: str,
    unique_id_prefix: str,
    runtime_data: dict[str, Any] | None = None,
) -> list["ADS1115Sensor"]:
    """Create ADS1115 entities from configuration."""
    if not LIBRARY_AVAILABLE:
        _LOGGER.error("Failed to import ads1x15 library. Make sure it's installed.")
        return []

    alock, created = get_i2c_bus_lock(hass, i2c_locks_key, bus)
    if created:
        _LOGGER.warning("ADS1115 created new lock for I2C bus %s", bus)

    try:
        adc = await async_i2c_call(hass, alock, ADS1x15.ADS1115, bus, address)
        await async_i2c_call(hass, alock, adc.setDataRate, adc.DR_ADS111X_128)
        await async_i2c_call(hass, alock, adc.setGain, float(gain))
    except Exception as ex:
        _LOGGER.error("Failed to initialize ADS1115: %s", ex)
        return []

    update_interval = timedelta(seconds=max(1, int(interval)))
    entities: list[ADS1115Sensor] = []
    for raw_channel_cfg in channels_config:
        channel_config = _normalize_channel_config(raw_channel_cfg)
        channel_number = channel_config[CONF_CHANNEL_NUMBER]
        channel_name = channel_config[CONF_NAME]

        entities.append(
            ADS1115Sensor(
                hass=hass,
                adc=adc,
                name=f"{name} {channel_name}",
                channel=channel_number,
                unit=channel_config[CONF_UNIT],
                min_val=channel_config[CONF_MIN],
                max_val=channel_config[CONF_MAX],
                scale=channel_config[CONF_SCALE],
                zero=channel_config[CONF_ZERO],
                use_filter=channel_config[CONF_FILTER],
                device_class=channel_config[CONF_CLASS],
                update_interval=update_interval,
                i2c_lock=alock,
                unique_id=f"{unique_id_prefix}_{channel_number}",
                bus=bus,
                address=address,
                device_name=name,
                runtime_data=runtime_data,
            )
        )
    return entities


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,  # noqa: ARG001
) -> None:
    """Set up ADS1115 from legacy YAML platform config."""
    entities = await _async_build_entities(
        hass,
        name=config[CONF_NAME],
        bus=config[CONF_I2C_BUS],
        address=config[CONF_ADDRESS],
        gain=float(config[CONF_GAIN]),
        interval=int(config[CONF_INTERVAL]),
        channels_config=config[CONF_CHANNELS],
        i2c_locks_key=config.get(CONF_I2C_LOCKS_KEY, DEFAULT_I2C_LOCKS_KEY),
        unique_id_prefix=f"ads1115_i2c_{config[CONF_I2C_BUS]}_{config[CONF_ADDRESS]}",
    )
    if entities:
        async_add_entities(entities, True)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ADS1115 from a config entry."""
    runtime_data = get_entry_runtime(hass, entry.entry_id)
    if runtime_data is None:
        runtime_data = ensure_entry_runtime(hass, entry)

    channels_config = entry.options.get(
        CONF_CHANNELS,
        entry.data.get(CONF_CHANNELS, [_default_channel_config(0)]),
    )
    if not channels_config:
        channels_config = [_default_channel_config(0)]

    entities = await _async_build_entities(
        hass,
        name=entry.data.get(CONF_NAME, DEFAULT_NAME),
        bus=int(entry.data.get(CONF_I2C_BUS, DEFAULT_I2C_BUS)),
        address=int(entry.data.get(CONF_ADDRESS, DEFAULT_I2C_ADDRESS)),
        gain=float(entry.options.get(CONF_GAIN, entry.data.get(CONF_GAIN, DEFAULT_GAIN))),
        interval=int(
            entry.options.get(CONF_INTERVAL, entry.data.get(CONF_INTERVAL, DEFAULT_INTERVAL))
        ),
        channels_config=channels_config,
        i2c_locks_key=entry.options.get(
            CONF_I2C_LOCKS_KEY,
            entry.data.get(CONF_I2C_LOCKS_KEY, DEFAULT_I2C_LOCKS_KEY),
        ),
        unique_id_prefix=f"ads1115_{entry.entry_id}",
        runtime_data=runtime_data,
    )
    if entities:
        async_add_entities(entities, True)


class ADS1115Sensor(SensorEntity):
    """Implementation of an ADS1115 ADC sensor."""

    def __init__(
        self,
        *,
        hass: HomeAssistant,
        adc,
        name: str,
        channel: int,
        unit: str,
        min_val: int,
        max_val: int,
        scale: int,
        zero: int,
        use_filter: bool,
        device_class: str | None,
        update_interval: timedelta,
        i2c_lock,
        unique_id: str,
        bus: int,
        address: int,
        device_name: str,
        runtime_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self._adc_device = adc
        self._channel = channel
        self._unit = unit
        self._min = min_val
        self._max = max_val
        self._scale = scale
        self._zero = zero
        self._filter_enabled = use_filter
        self._device_class_name = device_class
        self._state = None
        self._available = True
        self._update_interval_s = update_interval.total_seconds()
        self._runtime_data = runtime_data
        self._i2c_lock = i2c_lock
        self._last_update_s = 0.0

        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = unit
        self._attr_device_info = {
            "identifiers": {(DOMAIN, f"{bus}:{address}")},
            "name": f"{device_name} ({bus}:0x{address:02X})",
            "manufacturer": "Texas Instruments",
            "model": "ADS1115",
        }

        if self._device_class_name and hasattr(SensorDeviceClass, self._device_class_name.upper()):
            self._attr_device_class = getattr(SensorDeviceClass, self._device_class_name.upper())

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    @property
    def native_value(self):
        """Return the state of the sensor."""
        return self._state

    async def async_update(self) -> None:
        """Fetch new state data for the sensor."""
        if self._runtime_data is not None and not get_runtime_readout_enabled(self._runtime_data):
            return

        now = time.monotonic()
        update_interval_s = (
            float(get_runtime_interval(self._runtime_data, int(self._update_interval_s)))
            if self._runtime_data is not None
            else self._update_interval_s
        )
        if now - self._last_update_s < update_interval_s:
            return
        self._last_update_s = now

        try:
            raw = await async_i2c_call(
                self.hass,
                self._i2c_lock,
                self._adc_device.readADC,
                self._channel,
            )
            _LOGGER.debug(
                "Raw ADC value/voltage: %s/%s",
                raw,
                self._adc_device.toVoltage(raw),
            )
            self._state = self._adc_device.toVoltage(raw)
            self._available = True
        except Exception as ex:
            _LOGGER.error("Error reading ADS1115 channel %s: %s", self._channel, ex)
            self._available = False
