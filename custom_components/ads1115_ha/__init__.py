"""ADS1115 ADC Sensor integration for Home Assistant."""
from homeassistant.helpers.discovery import load_platform
import homeassistant.helpers.config_validation as cv
import voluptuous as vol

DOMAIN = "ads1115"

# This will load the sensor platform
async def async_setup(hass, config):
    """Set up the ADS1115 component."""
    if DOMAIN not in config:
        return True
    
    # Forward the config to the sensor platform
    hass.async_create_task(
        load_platform(hass, "sensor", DOMAIN, config[DOMAIN], config)
    )
    
    return True