"""Config flow for ADS1115 integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

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
    CONF_READOUT_ENABLED,
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
    DEFAULT_READOUT_ENABLED,
    DEFAULT_SCALE,
    DEFAULT_UNIT,
    DOMAIN,
)

_GAIN_LABEL_TO_VALUE = {
    "2/3": 2.0 / 3.0,
    "1": 1.0,
    "2": 2.0,
    "4": 4.0,
    "8": 8.0,
    "16": 16.0,
}

_GAIN_OPTIONS = list(_GAIN_LABEL_TO_VALUE.keys())
_CHANNEL_OPTIONS = [str(channel) for channel in range(4)]


def _gain_to_label(value: float) -> str:
    for label, gain_value in _GAIN_LABEL_TO_VALUE.items():
        if abs(float(value) - gain_value) < 0.0001:
            return label
    return "2"


def _selected_channels(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (int, str)):
        values = [raw_value]
    else:
        values = list(raw_value)
    return sorted({int(value) for value in values})


def _default_channel_config(channel_number: int) -> dict[str, Any]:
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


def _build_channel_configs(
    selected_channels: list[int],
    existing_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    existing_map: dict[int, dict[str, Any]] = {}
    for config in existing_configs or []:
        if not isinstance(config, dict):
            continue
        channel = config.get(CONF_CHANNEL_NUMBER)
        if channel is None:
            continue
        existing_map[int(channel)] = config

    channel_configs = []
    for channel in selected_channels:
        channel_configs.append(existing_map.get(channel, _default_channel_config(channel)))
    return channel_configs


def _normalize_import_channels(raw_channels: Any) -> list[dict[str, Any]]:
    if not raw_channels:
        return [_default_channel_config(0)]

    if isinstance(raw_channels, (list, tuple)):
        if all(isinstance(channel, dict) for channel in raw_channels):
            channel_dicts = [
                dict(channel)
                for channel in raw_channels
                if CONF_CHANNEL_NUMBER in channel
            ]
            if channel_dicts:
                return channel_dicts
        selected = _selected_channels(raw_channels)
        if selected:
            return _build_channel_configs(selected)

    return [_default_channel_config(0)]


def _user_schema(
    *,
    default_name: str = DEFAULT_NAME,
    default_bus: int = DEFAULT_I2C_BUS,
    default_address: int = DEFAULT_I2C_ADDRESS,
    default_gain_label: str = "2",
    default_interval: int = DEFAULT_INTERVAL,
    default_lock_key: str = DEFAULT_I2C_LOCKS_KEY,
    default_channels: list[int] | None = None,
) -> vol.Schema:
    channels_default = [str(channel) for channel in (default_channels if default_channels else [0])]
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=default_name): str,
            vol.Required(CONF_I2C_BUS, default=default_bus): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=9,
                    mode="box",
                    step=1,
                )
            ),
            vol.Required(CONF_ADDRESS, default=default_address): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=127,
                    mode="box",
                    step=1,
                )
            ),
            vol.Required(CONF_GAIN, default=default_gain_label): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=_GAIN_OPTIONS,
                    mode="dropdown",
                )
            ),
            vol.Required(CONF_INTERVAL, default=default_interval): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=3600,
                    mode="box",
                    step=1,
                )
            ),
            vol.Required(
                CONF_I2C_LOCKS_KEY,
                default=default_lock_key,
            ): str,
            vol.Required(CONF_CHANNELS, default=channels_default): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=_CHANNEL_OPTIONS,
                    mode="dropdown",
                    multiple=True,
                )
            ),
        }
    )


def _options_schema(
    *,
    default_gain_label: str,
    default_lock_key: str,
    default_channels: list[str],
) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_GAIN, default=default_gain_label): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=_GAIN_OPTIONS,
                    mode="dropdown",
                )
            ),
            vol.Required(
                CONF_I2C_LOCKS_KEY,
                default=default_lock_key,
            ): str,
            vol.Required(CONF_CHANNELS, default=default_channels): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=_CHANNEL_OPTIONS,
                    mode="dropdown",
                    multiple=True,
                )
            ),
        }
    )


class ADS1115ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow for ADS1115."""

    VERSION = 1

    async def async_step_import(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Import YAML configuration."""
        if user_input is None:
            return self.async_abort(reason="invalid_config")

        user_input = dict(user_input)
        if CONF_NAME not in user_input:
            user_input[CONF_NAME] = DEFAULT_NAME
        if CONF_I2C_BUS not in user_input:
            user_input[CONF_I2C_BUS] = DEFAULT_I2C_BUS
        if CONF_ADDRESS not in user_input:
            user_input[CONF_ADDRESS] = DEFAULT_I2C_ADDRESS
        if CONF_GAIN not in user_input:
            user_input[CONF_GAIN] = DEFAULT_GAIN
        if CONF_INTERVAL not in user_input:
            user_input[CONF_INTERVAL] = DEFAULT_INTERVAL
        if CONF_I2C_LOCKS_KEY not in user_input:
            user_input[CONF_I2C_LOCKS_KEY] = DEFAULT_I2C_LOCKS_KEY

        unique_id = f"{int(user_input[CONF_I2C_BUS])}:{int(user_input[CONF_ADDRESS])}"
        await self.async_set_unique_id(unique_id)
        self._abort_if_unique_id_configured()

        gain_raw = user_input[CONF_GAIN]
        gain = (
            _GAIN_LABEL_TO_VALUE[gain_raw]
            if isinstance(gain_raw, str) and gain_raw in _GAIN_LABEL_TO_VALUE
            else float(gain_raw)
        )
        channels_config = _normalize_import_channels(user_input.get(CONF_CHANNELS))

        return self.async_create_entry(
            title=f"{user_input[CONF_NAME]} ({int(user_input[CONF_I2C_BUS])}:0x{int(user_input[CONF_ADDRESS]):02X})",
            data={
                CONF_NAME: user_input[CONF_NAME],
                CONF_I2C_BUS: int(user_input[CONF_I2C_BUS]),
                CONF_ADDRESS: int(user_input[CONF_ADDRESS]),
            },
            options={
                CONF_GAIN: float(gain),
                CONF_INTERVAL: int(user_input[CONF_INTERVAL]),
                CONF_I2C_LOCKS_KEY: str(user_input[CONF_I2C_LOCKS_KEY]),
                CONF_CHANNELS: channels_config,
                CONF_READOUT_ENABLED: bool(
                    user_input.get(CONF_READOUT_ENABLED, DEFAULT_READOUT_ENABLED)
                ),
            },
        )

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle UI setup flow."""
        errors: dict[str, str] = {}

        if user_input is not None:
            channels = _selected_channels(user_input[CONF_CHANNELS])
            if not channels:
                errors["base"] = "channels_required"
            else:
                unique_id = f"{int(user_input[CONF_I2C_BUS])}:{int(user_input[CONF_ADDRESS])}"
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                gain = _GAIN_LABEL_TO_VALUE[str(user_input[CONF_GAIN])]
                return self.async_create_entry(
                    title=f"{user_input[CONF_NAME]} ({int(user_input[CONF_I2C_BUS])}:0x{int(user_input[CONF_ADDRESS]):02X})",
                    data={
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_I2C_BUS: int(user_input[CONF_I2C_BUS]),
                        CONF_ADDRESS: int(user_input[CONF_ADDRESS]),
                    },
                    options={
                        CONF_GAIN: float(gain),
                        CONF_INTERVAL: int(user_input[CONF_INTERVAL]),
                        CONF_I2C_LOCKS_KEY: str(user_input[CONF_I2C_LOCKS_KEY]),
                        CONF_CHANNELS: _build_channel_configs(channels),
                        CONF_READOUT_ENABLED: bool(
                            user_input.get(CONF_READOUT_ENABLED, DEFAULT_READOUT_ENABLED)
                        ),
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_user_schema(default_gain_label=_gain_to_label(DEFAULT_GAIN)),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry):
        """Get options flow handler."""
        return ADS1115OptionsFlow(config_entry)


class ADS1115OptionsFlow(config_entries.OptionsFlowWithConfigEntry):
    """Handle ADS1115 options flow."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage options."""
        errors: dict[str, str] = {}
        existing_channels = self.config_entry.options.get(CONF_CHANNELS, [])
        channel_numbers: set[int] = set()
        for channel_config in existing_channels:
            if isinstance(channel_config, dict):
                channel_value = channel_config.get(CONF_CHANNEL_NUMBER)
            else:
                channel_value = channel_config
            if channel_value is None:
                continue
            channel_numbers.add(int(channel_value))

        default_channels = sorted(channel_numbers) or [0]
        default_channel_values = [str(channel) for channel in default_channels]

        if user_input is not None:
            channels = _selected_channels(user_input[CONF_CHANNELS])
            if not channels:
                errors["base"] = "channels_required"
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_GAIN: float(_GAIN_LABEL_TO_VALUE[str(user_input[CONF_GAIN])]),
                        CONF_INTERVAL: int(
                            self.config_entry.options.get(
                                CONF_INTERVAL,
                                self.config_entry.data.get(CONF_INTERVAL, DEFAULT_INTERVAL),
                            )
                        ),
                        CONF_I2C_LOCKS_KEY: str(user_input[CONF_I2C_LOCKS_KEY]),
                        CONF_CHANNELS: _build_channel_configs(channels, existing_channels),
                        CONF_READOUT_ENABLED: bool(
                            self.config_entry.options.get(
                                CONF_READOUT_ENABLED,
                                self.config_entry.data.get(
                                    CONF_READOUT_ENABLED,
                                    DEFAULT_READOUT_ENABLED,
                                ),
                            )
                        ),
                    },
                )

        return self.async_show_form(
            step_id="init",
            data_schema=_options_schema(
                default_gain_label=_gain_to_label(
                    float(self.config_entry.options.get(CONF_GAIN, DEFAULT_GAIN))
                ),
                default_lock_key=str(
                    self.config_entry.options.get(
                        CONF_I2C_LOCKS_KEY,
                        DEFAULT_I2C_LOCKS_KEY,
                    )
                ),
                default_channels=default_channel_values,
            ),
            errors=errors,
        )
