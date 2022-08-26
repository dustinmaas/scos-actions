from dataclasses import dataclass
from typing import Optional

from scos_actions.metadata.annotation_segment import AnnotationSegment


@dataclass
class CalibrationAnnotation(AnnotationSegment):
    """
    Interface for generating CalibrationAnnotation segments.

    Most values are read from sensor and sigan calibrations,
    expected to exist in a ``measurement_result`` dictionary.
    The sensor and sigan calibration parameters are required.

    Refer to the documentation of the ``ntia-sensor`` extension of
    SigMF for more information.

    :param sigan_cal: Sigan calibration result, likely stored
        in the ``measurement_result`` dictionary. This should contain
        keys: ``gain_sigan``, ``noise_figure_sigan``, ``1db_compression_sigan``,
        and ``enbw_sigan``.
    :param sensor_cal: Sensor calibration result, likely stored
        in the ``measurement_result`` dictionary. This should contain
        keys: ``gain_preselector``, ``noise_figure_sensor``, ``1db_compression_sensor``,
        ``enbw_sensor``, and ``gain_sensor``. Optionally, it can also include
        a ``temperature`` key.
    :param mean_noise_power_sensor: Mean noise power density of the sensor.
    :param mean_noise_power_units: The units of ``mean_noise_power_sensor``.
    :param mean_noise_power_reference: Reference point for ``mean_noise_power_sensor``,
        e.g. ``"signal analyzer input"``, ``"preselector input"``, ``"antenna terminal"``.
    """

    sigan_cal = Optional[dict] = None
    sensor_cal = Optional[dict] = None
    mean_noise_power_sensor: Optional[float] = None
    mean_noise_power_units: Optional[str] = None
    mean_noise_power_reference: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Load values from sensor and sigan calibrations
        self.gain_sigan = self.sigan_cal["gain_sigan"]
        self.noise_figure_sigan = self.sigan_cal["noise_figure_sigan"]
        self.compression_point_sigan = self.sigan_cal["1db_compression_sigan"]
        self.enbw_sigan = self.sigan_cal["enbw_sigan"]
        self.gain_preselector = self.sensor_cal["gain_preselector"]
        self.noise_figure_sensor = self.sensor_cal["noise_figure_sensor"]
        self.compression_point_sensor = self.sensor_cal["1db_compression_sensor"]
        self.enbw_sensor = self.sensor_cal["enbw_sensor"]
        if "temperature" in self.sensor_cal:
            self.temperature = self.sensor_cal["temperature"]
        else:
            self.temperature = None
        # Additional key gain_sensor is not in SigMF ntia-sensor spec but is included
        self.gain_sensor = self.sensor_cal["gain_sensor"]
        # Define SigMF key names
        self.sigmf_keys.update(
            {
                "gain_sigan": "ntia-sensor:gain_sigan",
                "noise_figure_sigan": "ntia-sensor:noise_figure_sigan",
                "one_db_compression_point_sigan": "ntia-sensor:1db_compression_point_sigan",
                "enbw_sigan": "ntia-sensor:enbw_sigan",
                "gain_preselector": "ntia-sensor:gain_preselector",
                "gain_sensor": "ntia-sensor:gain_sensor",  # This is not a valid ntia-sensor key
                "noise_figure_sensor": "ntia-sensor:noise_figure_sensor",
                "one_db_compression_point_sensor": "ntia-sensor:1db_compression_point_sensor",
                "enbw_sensor": "ntia-sensor:enbw_sensor",
                "mean_noise_power_sensor": "ntia-sensor:mean_noise_power_sensor",
                "mean_noise_power_units": "ntia-sensor:mean_noise_power_units",
                "mean_noise_power_reference": "ntia-sensor:mean_noise_power_reference",
                "temperature": "ntia-sensor:temperature",
            }
        )
        # Create annotation segment
        super().create_annotation_segment()
