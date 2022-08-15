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
r"""Acquire a NASCTN SEA data product.

Currently in development.
"""
import logging
from time import perf_counter
from typing import Tuple

import numexpr as ne
import numpy as np
from scipy.signal import sosfilt

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.hardware import gps as mock_gps
from scos_actions.metadata.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.signal_processing.apd import get_apd
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_window,
    get_fft_window_correction,
)
from scos_actions.signal_processing.filtering import (
    generate_elliptic_iir_low_pass_filter,
)
from scos_actions.signal_processing.power_analysis import (
    apply_power_detector,
    calculate_power_watts,
    calculate_pseudo_power,
    create_power_detector,
    filter_quantiles,
)
from scos_actions.signal_processing.unit_conversion import (
    convert_linear_to_dB,
    convert_watts_to_dBm,
)

logger = logging.getLogger(__name__)

# Define parameter keys
IIR_APPLY = "iir_apply"
RP_DB = "iir_rp_dB"
RS_DB = "iir_rs_dB"
IIR_CUTOFF_HZ = "iir_cutoff_Hz"
IIR_WIDTH_HZ = "iir_width_Hz"
QFILT_APPLY = "qfilt_apply"
Q_LO = "qfilt_qlo"
Q_HI = "qfilt_qhi"
FFT_SIZE = "fft_size"
NUM_FFTS = "nffts"
FFT_WINDOW_TYPE = "fft_window_type"
APD_BIN_SIZE_DB = "apd_bin_size_dB"
TD_BIN_SIZE_MS = "td_bin_size_ms"
ROUND_TO = "round_to_places"
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"


class NasctnSeaDataProduct(Action):
    """Acquire a stepped-frequency NASCTN SEA data product.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)

        # Setup/pull config parameters
        # TODO: All parameters in this section should end up hard-coded
        # For now they are parameterized in the action config for testing
        self.iir_rp_dB = utils.get_parameter(RP_DB, self.parameters)
        self.iir_rs_dB = utils.get_parameter(RS_DB, self.parameters)
        self.iir_cutoff_Hz = utils.get_parameter(IIR_CUTOFF_HZ, self.parameters)
        self.iir_width_Hz = utils.get_parameter(IIR_WIDTH_HZ, self.parameters)
        self.qfilt_qlo = utils.get_parameter(Q_LO, self.parameters)
        self.qfilt_qhi = utils.get_parameter(Q_HI, self.parameters)
        self.fft_window_type = utils.get_parameter(FFT_WINDOW_TYPE, self.parameters)

        # TODO: These parameters should not be hard-coded
        # None of these should be lists - all single values
        # self.iir_apply = utils.get_parameter(IIR_APPLY, self.parameters)
        # self.qfilt_apply = utils.get_parameter(QFILT_APPLY, self.parameters)
        self.fft_size = utils.get_parameter(FFT_SIZE, self.parameters)
        # self.nffts = utils.get_parameter(NUM_FFTS, self.parameters)
        # self.apd_bin_size_dB = utils.get_parameter(APD_BIN_SIZE_DB, self.parameters)
        # self.td_bin_size_ms = utils.get_parameter(TD_BIN_SIZE_MS, self.parameters)
        # self.round_to = utils.get_parameter(ROUND_TO, self.parameters)
        self.sample_rate_Hz = utils.get_parameter(SAMPLE_RATE, self.parameters)

        # Construct IIR filter
        self.iir_sos = generate_elliptic_iir_low_pass_filter(
            self.iir_rp_dB,
            self.iir_rs_dB,
            self.iir_cutoff_Hz,
            self.iir_width_Hz,
            self.sample_rate_Hz,
        )

        # Generate FFT window and get its energy correction factor
        self.fft_window = get_fft_window(self.fft_window_type, self.fft_size)
        self.fft_window_ecf = get_fft_window_correction(self.fft_window, "energy")

        # Create power detectors
        self.fft_detector = create_power_detector("FftMeanMaxDetector", ["mean", "max"])
        self.td_detector = create_power_detector("TdMeanMaxDetector", ["mean", "max"])

    def __call__(self, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        # Temporary: remove config parameters which will be hard-coded eventually
        for key in [
            RP_DB,
            RS_DB,
            IIR_CUTOFF_HZ,
            IIR_WIDTH_HZ,
            Q_LO,
            Q_HI,
            FFT_WINDOW_TYPE,
        ]:
            self.parameters.pop(key)
        self.test_required_components()

        iteration_params = utils.get_iterable_parameters(self.parameters)

        # TODO:
        # For now, this iterates (capture IQ -> process data product) for each
        # configured frequency. It is probably better to do all IQ captures first,
        # then generate all data products, or to parallelize captures/processing.

        start_action = perf_counter()
        for i, p in enumerate(iteration_params, start=1):
            logger.debug(f"Generating data product for parameters: {p}")
            # Capture IQ data
            measurement_result = self.capture_iq(schedule_entry, task_id, i, p)
            # Generate data product, overwrite IQ data
            measurement_result["data"] = self.generate_data_product(
                measurement_result, p
            )

            # Send signal
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=measurement_result["data"],
                metadata=None,  # TODO: Add metadata
            )
        action_done = perf_counter()
        logger.debug(
            f"IQ Capture and data processing completed in {action_done-start_action:.2f}"
        )

    def capture_iq(self, schedule_entry, task_id, recording_id, params) -> dict:
        start_time = utils.get_datetime_str_now()
        tic = perf_counter()
        # Configure signal analyzer + preselector
        self.configure(params)
        # Get IQ capture parameters
        sample_rate = self.sigan.sample_rate
        duration_ms = utils.get_parameter(DURATION_MS, params)
        nskip = utils.get_parameter(NUM_SKIP, params)
        num_samples = int(sample_rate * duration_ms * 1e-3)
        # Collect IQ data
        # measurement_result = super().acquire_data(num_samples, nskip)
        measurement_result = self.sigan.acquire_time_domain_samples(num_samples, nskip)
        end_time = utils.get_datetime_str_now()
        # Store some metadata with the IQ
        measurement_result.update(params)
        measurement_result["start_time"] = start_time
        measurement_result["end_time"] = end_time
        measurement_result["domain"] = Domain.TIME.value
        measurement_result["measurement_type"] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result["task_id"] = task_id
        toc = perf_counter()
        logger.debug(f"IQ Capture ({duration_ms} ms) completed in {toc-tic:.2f} s.")
        return measurement_result

    def generate_data_product(
        self, measurement_result: dict, params: dict
    ) -> np.ndarray:
        # Load IQ, process, return data product, for single channel
        # TODO: Explore parallelizing computation tasks
        logger.debug(f'Generating data product for {measurement_result["task_id"]}')
        iq = measurement_result["data"].astype(np.complex128)
        data_product = []

        # Get FFT amplitudes using unfiltered data
        logger.debug("Getting FFT results...")
        tic = perf_counter()
        data_product.extend(self.get_fft_results(iq, params))
        toc = perf_counter()
        logger.debug(f"FFT computation complete in {toc-tic:.2f} s")

        # Filter IQ data
        if params[IIR_APPLY]:
            logger.debug(f"Applying IIR low-pass filter to IQ data...")
            tic = perf_counter()
            iq = sosfilt(self.iir_sos, iq)
            toc = perf_counter()
            logger.debug(f"IIR filter applied to IQ samples in {toc-tic:.2f} s")
        else:
            logger.debug(f"Skipping IIR filtering of IQ data...")

        logger.debug("Calculating time-domain power statistics...")
        tic = perf_counter()
        data_product.extend(self.get_td_power_results(iq, params))
        toc = perf_counter()
        logger.debug(f"Time domain power statistics calculated in {toc-tic:.2f} s")

        logger.debug("Generating APD...")
        tic = perf_counter()
        data_product.extend(self.get_apd_results(iq, params))
        toc = perf_counter()
        logger.debug(f"APD result generated in {toc-tic:.2f} s")

        del iq

        # Quantize power results
        tic = perf_counter()
        for i, data in enumerate(data_product):
            if i == 4:
                # Do not round APD probability axis
                continue
            data.round(decimals=params[ROUND_TO], out=data)
        toc = perf_counter()
        logger.debug(
            f"Data product rounded to {params[ROUND_TO]} decimal places in {toc-tic:.2f} s"
        )

        # Reduce data types to half-precision floats
        tic = perf_counter()
        for i in range(len(data_product)):
            data_product[i] = data_product[i].astype(np.half)
        toc = perf_counter()
        logger.debug(f"Reduced data types to half-precision float in {toc-tic:.2f} s")

        return np.array(data_product, dtype=object)

    def get_fft_results(
        self, iqdata: np.ndarray, params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        # IQ data already scaled for calibrated gain
        fft_result = get_fft(
            time_data=iqdata,
            fft_size=params[FFT_SIZE],
            norm="forward",
            fft_window=self.fft_window,
            num_ffts=params[NUM_FFTS],
            shift=False,
            workers=1,  # TODO: Configure for parallelization
        )
        fft_result = calculate_pseudo_power(fft_result)
        fft_result = apply_power_detector(
            fft_result, self.fft_detector
        )  # First array is mean, second is max
        ne.evaluate("fft_result/50", out=fft_result)  # Finish conversion to Watts
        # Shift frequencies of reduced result
        fft_result = np.fft.fftshift(fft_result, axes=(1,))
        fft_result = convert_watts_to_dBm(fft_result)
        fft_result -= 3  # Baseband/RF power conversion
        fft_result -= 10.0 * np.log10(
            self.sample_rate_Hz
        )  # PSD scaling # TODO: Assure this is the correct sample rate
        fft_result += 20.0 * np.log10(self.fft_window_ecf)  # Window energy correction
        return fft_result[0], fft_result[1]

    def get_apd_results(
        self, iqdata: np.ndarray, params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        p, a = get_apd(iqdata, params[APD_BIN_SIZE_DB])
        # Convert dBV to dBm:
        # a = a * 2 : dBV --> dB(V^2)
        # a = a - impedance_dB : dB(V^2) --> dBW
        # a = a + 27 : dBW --> dBm (+30) and RF/baseband conversion (-3)
        scale_factor = 27 - convert_linear_to_dB(50.0)  # Hard-coded for 50 Ohms.
        ne.evaluate("(a*2)+scale_factor", out=a)
        return p, a

    def get_td_power_results(
        self, iqdata: np.ndarray, params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Reshape IQ data into blocks
        block_size = int(
            params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3
        )  # TODO: Assure this uses correct sample rate
        n_blocks = len(iqdata) // block_size
        iqdata = iqdata.reshape(n_blocks, block_size)

        iq_pwr = calculate_power_watts(iqdata, impedance_ohms=50.0)

        if params[QFILT_APPLY]:
            # Apply quantile filtering before computing power statistics
            logger.info("Quantile-filtering time domain power data...")
            iq_pwr = filter_quantiles(iq_pwr, self.qfilt_qlo, self.qfilt_qhi)
            # Diagnostics
            num_nans = np.count_nonzero(np.isnan(iq_pwr))
            nan_pct = num_nans * 100 / len(iq_pwr.flatten())
            logger.debug(
                f"Rejected {num_nans} samples ({nan_pct:.2f}% of total capture)"
            )
        else:
            logger.info("Quantile-filtering disabled. Skipping...")

        # Apply mean/max detectors
        td_result = apply_power_detector(iq_pwr, self.td_detector, ignore_nan=True)

        # Convert to dBm
        td_result = convert_watts_to_dBm(td_result)

        # Account for RF/baseband power difference
        td_result -= 3

        return td_result[0], td_result[1]

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            raise RuntimeError(msg)
        # TODO: Add additional health checks
        return None

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        # TODO (low-priority)
        return __doc__

    def get_sigmf_builder(self, measurement_result) -> SigMFBuilder:
        # TODO (low-priority)
        # Create metadata annotations for the data
        return None

    def is_complex(self) -> bool:
        return False
