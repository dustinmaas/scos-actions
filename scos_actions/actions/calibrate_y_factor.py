# What follows is a parameterizable description of the algorithm used by this
# action. The first line is the summary and should be written in plain text.
# Everything following that is the extended description, which can be written
# in Markdown and MathJax. Each name in curly brackets '{}' will be replaced
# with the value specified in the `description` method which can be found at
# the very bottom of this file. Since this parameterization step affects
# everything in curly brackets, math notation such as {m \over n} must be
# escaped to {{m \over n}}.
#
# To print out this docstring after parameterization, see
# scos-sensor/scripts/print_action_docstring.py. You can then paste that into the
# SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Perform a Y-Factor Calibration.
Supports calibration of gain and noise figure for one or more channels.
For each center frequency, sets the preselector to the noise diode path, turns
noise diode on, performs a mean power measurement, turns the noise diode off and
performs another mean power measurement. The mean power on and mean power off
data are used to compute the noise figure and gain. Mean power is calculated in the
time domain{filtering_suffix}.

# {name}

## Signal analyzer setup and sample acquisition

Each time this task runs, the following process is followed to take measurements
separately with the noise diode off and on:
{acquisition_plan}

## Time-domain processing

{filtering_description}

Next, mean power calculations are performed. Sample amplitudes are divided by two
to account for the power difference between RF and complex baseband samples. Then,
power is calculated element-wise from the complex time-domain samples. The power of
each sample is defined by the square of the magnitude of the complex sample, divided by
the system impedance, which is taken to be 50 Ohms.

## Y-Factor Method

The mean power for the noise diode on and off captures are calculated by taking the
mean of each array of power samples. Next, the Y-factor is calculated by:

$$ y = P_{{on}} / P_{{off}} $$

Where $P_{{on}}$ is the mean power measured with the noise diode on, and $P_{{off}}$
is the mean power measured with the noise diode off. The linear noise factor is then
calculated by:

$$ NF = \frac{{ENR}}{{y - 1}} $$

Where $ENR$ is the excess noise ratio, in linear units, of the noise diode used for
the power measurements. Next, the linear gain is calculated by:

$$ G = \frac{{P_{{on}}}}{{k_B T B_{{eq}} (ENR + NF)}} $$

Where $k_B$ is Boltzmann's constant, $T$ is the calibration temperature in Kelvins,
and $B_{{eq}}$ is the sensor's equivalent noise bandwidth. Finally, the noise factor
and linear gain are converted to noise figure $F_N$ and decibel gain $G_{{dB}}$:

$$ G_{{dB}} = 10 \log_{{10}}(G) $$
$$ F_N = 10 \log_{{10}}(NF) $$
"""

import logging
import time
import numpy as np

from scipy.constants import Boltzmann
from scipy.signal import sosfilt

from scos_actions import utils
from scos_actions.hardware import gps as mock_gps
from scos_actions.settings import sensor_calibration
from scos_actions.settings import SENSOR_CALIBRATION_FILE
from scos_actions.actions.interfaces.action import Action
from scos_actions.utils import ParameterException, get_parameter

from scos_actions.signal_processing.calibration import (
    get_linear_enr,
    get_temperature,
    y_factor,
)
from scos_actions.signal_processing.power_analysis import (
    calculate_power_watts,
    create_power_detector,
)

from scos_actions.signal_processing.unit_conversion import convert_watts_to_dBm
from scos_actions.signal_processing.filtering import generate_elliptic_iir_low_pass_filter

import os

logger = logging.getLogger(__name__)

RF_PATH = 'rf_path'
NOISE_DIODE_ON = {RF_PATH: 'noise_diode_on'}
NOISE_DIODE_OFF = {RF_PATH: 'noise_diode_off'}

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"
IIR_APPLY = 'iir_apply'
IIR_RP = 'iir_rp_dB'
IIR_RS = 'iir_rs_dB'
IIR_CUTOFF = 'iir_cutoff_Hz'
IIR_WIDTH = 'iir_width_Hz'
CAL_SOURCE_IDX = 'cal_source_idx'
TEMP_SENSOR_IDX = 'temp_sensor_idx'


class YFactorCalibration(Action):
    """Perform a single- or stepped-frequency Y-factor calibration.

    The action will set any matching attributes found in the signal
    analyzer object. The following parameters are required by the
    action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector

    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer
    being used.

    :param parameters: The dictionary of parameters needed for the
        action and the signal analyzer.
    :param sigan: instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        logger.debug('Initializing calibration action')
        super().__init__(parameters, sigan, gps)
        self.sigan = sigan
        self.iteration_params = utils.get_iterable_parameters(parameters)
        self.power_detector = create_power_detector("MeanDetector", ["mean"])

        # IIR Filter Setup
        try:
            self.iir_apply = get_parameter(IIR_APPLY, parameters)
        except ParameterException:
            logger.info("Config parameter 'iir_apply' not provided. "
                        + "No IIR filtering will be used during calibration.")
            self.iir_apply = False

        if isinstance(self.iir_apply, list):
            raise ParameterException("Only one set of IIR filter parameters may be specified.")
        
        if self.iir_apply is True:
            self.iir_rp_dB = get_parameter(IIR_RP, parameters)
            self.iir_rs_dB = get_parameter(IIR_RS, parameters)
            self.iir_cutoff_Hz = get_parameter(IIR_CUTOFF, parameters)
            self.iir_width_Hz = get_parameter(IIR_WIDTH, parameters)
            self.sample_rate = get_parameter(SAMPLE_RATE, parameters)
            if not any([isinstance(v, list) for v in [self.iir_rp_dB, self.iir_rs_dB, self.iir_cutoff_Hz, self.iir_width_Hz, self.sample_rate]]):
                # Generate single filter ahead of calibration loop
                self.iir_sos = generate_elliptic_iir_low_pass_filter(
                    self.iir_rp_dB, self.iir_rs_dB, self.iir_cutoff_Hz, self.iir_width_Hz, self.sample_rate
                )
            else:
                raise ParameterException("Only one set of IIR filter parameters may be specified (including sample rate).")

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        detail = ''
        
        # Run calibration routine
        for i, p in enumerate(self.iteration_params):
            if i == 0:
                detail += self.calibrate(p)
            else:
                detail += os.linesep + self.calibrate(p)
        return detail

    def calibrate(self, params):
        # Configure signal analyzer
        sigan_params = params.copy()
        # Suppress warnings during sigan configuration
        for k in [DURATION_MS, NUM_SKIP, IIR_APPLY, IIR_RP, IIR_RS, IIR_CUTOFF, IIR_WIDTH, CAL_SOURCE_IDX, TEMP_SENSOR_IDX]:
            try:
                sigan_params.pop(k)
            except KeyError:
                continue
        # sigan_params also used as calibration args for getting sensor ENBW
        # if no IIR filtering is applied.
        #### TESTING #####
        sigan_params.pop("preamp_enable")
        super().configure_sigan(params["preamp_enable"])
        ####
        super().configure_sigan(sigan_params)

        # Get parameters from action config
        cal_source_idx = get_parameter(CAL_SOURCE_IDX, params)
        temp_sensor_idx = get_parameter(TEMP_SENSOR_IDX, params)
        sample_rate = get_parameter(SAMPLE_RATE, params)
        duration_ms = get_parameter(DURATION_MS, params)
        num_samples = int(sample_rate * duration_ms * 1e-3)
        nskip = get_parameter(NUM_SKIP, params)
        
        # Set noise diode on
        logger.debug('Setting noise diode on')
        super().configure_preselector(NOISE_DIODE_ON)
        time.sleep(1)

        # Get noise diode on IQ
        logger.debug("Acquiring IQ samples with noise diode ON")
        noise_on_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        sample_rate = noise_on_measurement_result["sample_rate"]

        # Set noise diode off
        logger.debug('Setting noise diode off')
        self.configure_preselector(NOISE_DIODE_OFF)
        time.sleep(1)

        # Get noise diode off IQ
        logger.debug('Acquiring IQ samples with noise diode OFF')
        noise_off_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        assert sample_rate == noise_off_measurement_result["sample_rate"], "Sample rate mismatch"

        # Apply IIR filtering to both captures if configured
        if self.iir_apply:
            cutoff_Hz = self.iir_cutoff_Hz
            width_Hz = self.iir_width_Hz
            enbw_hz = (cutoff_Hz + width_Hz) * 2.  # Roughly based on IIR filter
            logger.debug("Applying IIR filter to IQ captures")
            noise_on_data = sosfilt(self.iir_sos, noise_on_measurement_result["data"])
            noise_off_data = sosfilt(self.iir_sos, noise_off_measurement_result["data"])
        else:
            logger.debug('Skipping IIR filtering')
            # Get ENBW from sensor calibration
            cal_args = [sigan_params[k] for k in sensor_calibration.calibration_parameters]
            self.sigan.recompute_calibration_data(cal_args)
            enbw_hz = self.sigan.sensor_calibration_data["enbw_sensor"]
            noise_on_data = noise_on_measurement_result["data"]
            noise_off_data = noise_off_measurement_result["data"]

        # Get power values in time domain (division by 2 for RF/baseband conversion)
        pwr_on_watts = calculate_power_watts(noise_on_data / 2.)
        pwr_off_watts = calculate_power_watts(noise_off_data / 2.)

        # Y-Factor
        enr_linear = get_linear_enr(cal_source_idx)
        temp_k, temp_c, _ = get_temperature(temp_sensor_idx)
        noise_figure, gain = y_factor(
            pwr_on_watts, pwr_off_watts, enr_linear, enbw_hz, temp_k
        )

        # Update sensor calibration with results
        sensor_calibration.update(
            params,
            utils.get_datetime_str_now(),
            gain,
            noise_figure,
            temp_c,
            SENSOR_CALIBRATION_FILE,
        )

        # Debugging
        noise_floor_dBm = convert_watts_to_dBm(Boltzmann * temp_k * enbw_hz)
        logger.debug(f'Noise floor: {noise_floor_dBm:.2f} dBm')
        logger.debug(f'Noise figure: {noise_figure:.2f} dB')
        logger.debug(f"Gain: {gain:.2f} dB")
        
        # Detail results contain only FFT version of result for now
        return 'Noise Figure: {}, Gain: {}'.format(noise_figure, gain)

    @property
    def description(self):
        # Get parameters; they may be single values or lists
        frequencies = get_parameter(FREQUENCY, self.parameters)
        duration_ms = get_parameter(DURATION_MS, self.parameters)
        sample_rate = get_parameter(SAMPLE_RATE, self.parameters)

        if isinstance(duration_ms, list) and not isinstance(sample_rate, list):
            sample_rate = sample_rate * np.ones_like(duration_ms)
            sample_rate = sample_rate
        elif isinstance(sample_rate, list) and not isinstance(duration_ms, list):
            duration_ms = duration_ms * np.ones_like(sample_rate)
            duration_ms = duration_ms

        num_samples = duration_ms *  sample_rate * 1e-3

        if isinstance(num_samples, np.ndarray) and len(num_samples) != 1:
            num_samples = num_samples.tolist()
        else:
            num_samples = int(num_samples)

        if self.iir_apply is True:
            pb_edge = self.iir_cutoff_Hz / 1e6
            sb_edge = (self.iir_cutoff_Hz + self.iir_width_Hz) / 1e6
            filtering_suffix = ", after applying an IIR lowpass filter to the complex time-domain samples"
            filter_description = (
                """
                ### Filtering
                The acquired samples are then filtered using an elliptic IIR filter before
                performing the rest of the time-domain Y-factor calculations. The filter
                design produces the lowest order digital filter which loses no more than
                {self.iir_rp_dB} dB in the passband and has at least {self.iir_rs_dB} dB attenuation
                in the stopband. The filter has a defined passband edge at {pb_edge} MHz
                and a stopband edge at {sb_edge} MHz. From this filter design, second-order filter
                coefficients are generated in order to minimize numerical precision errors
                when filtering the time domain samples. The filtering function is implemented
                as a series of second-order filters with direct-form II transposed structure.

                ### Power Calculation
                """
            )
        else:
            filtering_suffix = ""
            filter_description = ""

        acquisition_plan = ""
        acq_plan_template = "The signal analyzer is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        acq_plan_template += "{parameters}"
        acq_plan_template += "Then, acquire samples for {duration_ms} ms.\n"

        used_keys = [FREQUENCY, DURATION_MS, "name"]
        for params in self.iteration_params:
            parameters = ""
            for name, value in params.items():
                if name not in used_keys:
                    parameters += f"{name} = {value}\n"
            acquisition_plan += acq_plan_template.format(
                **{
                    "center_frequency": params[FREQUENCY] / 1e6,
                    "parameters": parameters,
                    "duration_ms": params[DURATION_MS],
                }
            )
        
        definitions = {
            "name": self.name,
            "filtering_suffix": filtering_suffix,
            "filtering_description": filter_description,
            "acquisition_plan": acquisition_plan,
        }
        # __doc__ refers to the module docstring at the top of the file
        return __doc__ .format(**definitions)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)


