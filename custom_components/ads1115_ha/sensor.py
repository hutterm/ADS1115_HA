"""ADS1115 ADC Sensor integration for Home Assistant."""
import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import voluptuous as vol

from homeassistant.components.sensor import (
    PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import (
    CONF_ADDRESS,
    CONF_NAME,
    PERCENTAGE,
    UnitOfElectricPotential,
)
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

# Try to import the ADS1x15-ADC library
try:
    import ADS1x15 #import ads1115, ads1015, analogIn
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

# Configuration constants
CONF_I2C_BUS = "i2c_bus"
CONF_GAIN = "gain"
CONF_INTERVAL = "interval"
CONF_CHANNELS = "channels"
CONF_CHANNEL_NUMBER = "channel_number"
CONF_UNIT = "unit"
CONF_MIN = "min"
CONF_MAX = "max"
CONF_SCALE = "scale"
CONF_ZERO = "zero"
CONF_FILTER = "filter"
CONF_CLASS = "class"

DEFAULT_NAME = "ADS1115"
DEFAULT_I2C_ADDRESS = 0x48
DEFAULT_I2C_BUS = 1
DEFAULT_GAIN = 2
DEFAULT_INTERVAL = 1
DEFAULT_MIN = 0
DEFAULT_MAX = 65535
DEFAULT_SCALE = 65535
DEFAULT_UNIT = PERCENTAGE

# Voluptuous schemas
CHANNEL_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CHANNEL_NUMBER): vol.All(vol.Coerce(int), vol.Range(min=0, max=3)),
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_UNIT, default=DEFAULT_UNIT): cv.string,
        vol.Optional(CONF_MIN, default=DEFAULT_MIN): vol.Coerce(int),
        vol.Optional(CONF_MAX, default=DEFAULT_MAX): vol.Coerce(int),
        vol.Optional(CONF_SCALE, default=DEFAULT_SCALE): vol.Coerce(int),
        vol.Optional(CONF_ZERO): vol.Coerce(int),
        vol.Optional(CONF_FILTER, default=False): cv.boolean,
        vol.Optional(CONF_CLASS): cv.string,
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_I2C_BUS, default=DEFAULT_I2C_BUS): vol.Coerce(int),
        vol.Optional(CONF_ADDRESS, default=DEFAULT_I2C_ADDRESS): vol.Coerce(int),
        vol.Optional(CONF_GAIN, default=DEFAULT_GAIN): vol.All(
            vol.Coerce(int), vol.In([2/3, 1, 2, 4, 8, 16])
        ),
        vol.Optional(CONF_INTERVAL, default=DEFAULT_INTERVAL): vol.All(
            vol.Coerce(int), vol.Range(min=1)
        ),
        vol.Required(CONF_CHANNELS): vol.All(cv.ensure_list, [CHANNEL_SCHEMA]),
    }
)

# Simple Kalman filter implementation (since kalmanjs is used in the JS version)
class KalmanFilter:
    """Simple Kalman filter for smoothing sensor readings."""

    # Class-level constants for process and measurement noise
    DEFAULT_R = 0.01
    DEFAULT_Q = 3.0

    def __init__(self, r=DEFAULT_R, q=DEFAULT_Q):
        """Initialize the filter with process and measurement noise."""
        self.r = r  # Measurement noise
        self.q = q  # Process noise
        self.p = 1.0  # Initial error covariance
        self.x = 0.0  # Initial value

    def filter(self, measurement):
        """Apply Kalman filter to measurement."""
        # Prediction
        p = self.p + self.q

        # Update
        k = p / (p + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p

        return self.x


async def async_setup_platform(
    hass, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the ADS1115 sensor platform."""
    if not LIBRARY_AVAILABLE:
        _LOGGER.error("Failed to import ads1x15 library. Make sure it's installed.")
        return

    name = config[CONF_NAME]
    bus = config[CONF_I2C_BUS]
    address = config[CONF_ADDRESS]
    gain = config[CONF_GAIN]
    interval = config[CONF_INTERVAL]
    channels_config = config[CONF_CHANNELS]

    try:
        # Create ADC object
        adc = ADS1x15.ADS1115(bus, address)
        #adc.setDataRate(128)  # Default data rate
        adc.setDataRate(adc.DR_ADS111X_128)
        #adc.setMode(adc.MODE_CONTINUOUS)  # Continuous conversion mode
        adc.setGain(gain)
        #adc.requestADC(0)  
    except Exception as ex:
        _LOGGER.error("Failed to initialize ADS1115: %s", ex)
        return

    update_interval = timedelta(seconds=interval)
    entities = []

    # Create sensor entities for each configured channel
    for channel_config in channels_config:
        channel_number = channel_config[CONF_CHANNEL_NUMBER]
        channel_name = channel_config.get(CONF_NAME, f"ADC{channel_number}")
        unit = channel_config[CONF_UNIT]
        min_val = channel_config[CONF_MIN]
        max_val = channel_config[CONF_MAX]
        scale = channel_config[CONF_SCALE]
        zero = channel_config.get(CONF_ZERO, min_val)
        use_filter = channel_config[CONF_FILTER]
        device_class = channel_config.get(CONF_CLASS)

        # Ensure zero is within min/max bounds
        zero = max(min_val, min(max_val, zero))

        entities.append(
            ADS1115Sensor(
                adc,
                f"{name} {channel_name}",
                channel_number,
                unit,
                min_val,
                max_val,
                scale,
                zero,
                use_filter,
                device_class,
                update_interval,
                f"ads1115_i2c_{bus}_{address}_{channel_number}",
            )
        )

    async_add_entities(entities, True)


class ADS1115Sensor(SensorEntity):
    """Implementation of an ADS1115 ADC sensor."""

    def __init__(
        self,
        adc,
        name,
        channel,
        unit,
        min_val,
        max_val,
        scale,
        zero,
        use_filter,
        device_class,
        update_interval,
        unique_id,
    ):
        """Initialize the sensor."""
        self._adc_device = adc
        self._name = name
        self._channel = channel
        self._unit = unit
        self._min = min_val
        self._max = max_val
        self._scale = scale
        if zero < min_val:
            raise ValueError(f"Invalid configuration: zero ({zero}) must be greater than or equal to min_val ({min_val}).")
        self._zero = zero
        self._positive = max_val - zero
        self._negative = zero - min_val
        self._positive = max_val - zero
        self._negative = zero - min_val
        self._filter = KalmanFilter(r=0.01, q=3.0) if use_filter else None
        self._device_class = device_class
        self._state = None
        self._available = True
        self._update_interval = update_interval
        self._last_update = None
        self._attr_unique_id = unique_id

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def state(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def available(self):
        """Return True if entity is available."""
        return self._available

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        return self._unit

    @property
    def device_class(self):
        """Return the device class of this entity, if any."""
        if self._device_class and hasattr(SensorDeviceClass, self._device_class.upper()):
            return getattr(SensorDeviceClass, self._device_class.upper())
        return None

    @property
    def state_class(self):
        """Return the state class of this entity."""
        return SensorStateClass.MEASUREMENT

    @Throttle(timedelta(seconds=1))
    async def async_update(self):
        """Fetch new state data for the sensor."""
        try:
            raw = self._adc_device.readADC(self._channel)
            #print("{0:.3f} V".format(ADS.toVoltage(raw)))
            _LOGGER.debug("Raw ADC value/voltage: %s/%s", raw, self._adc_device.toVoltage(raw))

            self._state = self._adc_device.toVoltage(raw)
            
            # # Apply Kalman filter if configured
            # if self._filter:
            #     raw = self._filter.filter(raw)
            
            # # Constrain the value to min/max range
            # raw = max(self._min, min(self._max, raw))
            
            # # Calculate scaled value using the same approach as the JS version
            # negative = raw < self._zero
            # if negative:
            #     result = (raw - self._zero) / self._negative
            # else:
            #     result = (raw - self._zero) / self._positive
            
            # if negative:
            #     result = -1 - result
            
            # # Scale and round to get final value
            # self._state = round(result * self._scale)
            self._available = True
            
        except Exception as ex:
            _LOGGER.error("Error reading ADS1115 channel %s: %s", self._channel, ex)
            self._available = False