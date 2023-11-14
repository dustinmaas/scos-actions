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
r"""Capture time-domain IQ samples at the following {num_center_frequencies} frequencies: {center_frequencies}.

# {name}

## Signal Analyzer setup and sample acquisition

Each time this task runs, the following process is followed:

{acquisition_plan}

This will take a minimum of {min_duration_ms:.2f} ms, not including signal analyzer
tuning, dropping samples after retunes, and data storage.

## Time-domain processing

If specified, a voltage scaling factor is applied to the complex time-domain
signals.
"""

import logging

import numpy as np

from scos_actions import utils
from scos_actions.actions.acquire_single_freq_fft import (
    FREQUENCY,
    SAMPLE_RATE,
    NUM_SKIP,
    NUM_FFTS,
    FFT_SIZE,
    CLASSIFICATION,
    CAL_ADJUST,
    SingleFrequencyFftAcquisition,
)
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.metadata.structs import ntia_sensor
from scos_actions.metadata.structs.capture import CaptureSegment
from scos_actions.signals import measurement_action_completed
from scos_actions.utils import get_parameter

logger = logging.getLogger(__name__)


class SteppedFrequencyFftAcquisition(SingleFrequencyFftAcquisition):
    """Acquire IQ data at each of the requested frequencies.

    The action will set any matching attributes found in the
    signal analyzer object. The following parameters are required
    by the action:

        name: The name of the action.
        frequency: An iterable of center frequencies, in Hz.
        duration_ms: An iterable of measurement durations, per
            center_frequency, in ms

    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer
    being used.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: instance of SignalAnalyzerInterface
    """

    def __init__(self, parameters, sigan, gps=None):
        if gps is None:
            gps = MockGPS()
        super().__init__(parameters=parameters, sigan=sigan, gps=gps)
        num_center_frequencies = len(parameters[FREQUENCY])

        # Create iterable parameter set
        self.iterable_params = utils.get_iterable_parameters(parameters)

        self.sigan = sigan  # make instance variable to allow mocking
        self.num_center_frequencies = num_center_frequencies

    def __call__(self, schedule_entry: dict, task_id: int):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        saved_samples = 0

        for recording_id, measurement_params in enumerate(
            self.iterable_params, start=1
        ):
            self.get_sigmf_builder(schedule_entry)
            self.configure(measurement_params)
            fft_size = get_parameter(FFT_SIZE, measurement_params)
            nskip = get_parameter(NUM_SKIP, measurement_params)
            nffts = get_parameter(NUM_FFTS, measurement_params)
            cal_adjust = get_parameter(CAL_ADJUST, measurement_params)
            sample_rate = self.sigan.sample_rate
            num_samples = nffts * fft_size
            measurement_result = super().acquire_data(num_samples, nskip, cal_adjust)
            measurement_result.update(measurement_params)
            end_time = utils.get_datetime_str_now()
            measurement_result["end_time"] = end_time
            measurement_result["task_id"] = task_id
            measurement_result["name"] = self.name
            measurement_result["classification"] = self.classification
            m4s_result = self.apply_m4s(measurement_result)
            measurement_result.update(self.parameters)
            sigan_settings = self.get_sigan_settings(measurement_result)
            duration_ms = int(num_samples / sample_rate * 1e3)
            capture_segment = CaptureSegment(
                sample_start=0,
                global_index=saved_samples,
                frequency=measurement_params[FREQUENCY],
                datetime=measurement_result["capture_time"],
                duration=duration_ms,
                overload=measurement_result["overload"],
                sigan_settings=sigan_settings,
            )
            measurement_result["capture_segment"] = capture_segment

            self.create_metadata(measurement_result, recording_id)
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=measurement_result["data"],
                metadata=self.sigmf_builder.metadata,
            )
            saved_samples += len(m4s_result.ravel())

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""

        acquisition_plan = ""
        used_keys = [FREQUENCY, NUM_FFTS, FFT_SIZE, "name"]
        acq_plan_template = "The signal analyzer is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        acq_plan_template += "{parameters}"
        acq_plan_template += "Then, acquire samples.\n"

        duration_ms = self.fft_size * self.nffts / self.sigan.sample_rate * 1e3
        for measurement_params in self.iterable_params:
            parameters = ""
            for name, value in measurement_params.items():
                if name not in used_keys:
                    parameters += f"{name} = {value}\n"
            acquisition_plan += acq_plan_template.format(
                **{
                    "center_frequency": measurement_params[FREQUENCY] / 1e6,
                    "parameters": parameters,
                    "duration_ms": duration_ms,
                }
            )

        durations = [duration_ms for v in self.iterable_params]
        min_duration_ms = np.sum(durations)

        defs = {
            "name": self.name,
            "num_center_frequencies": self.num_center_frequencies,
            "center_frequencies": ", ".join(
                [f"{param[FREQUENCY] / 1e6:.2f} MHz" for param in self.iterable_params]
            ),
            "acquisition_plan": acquisition_plan,
            "min_duration_ms": min_duration_ms,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
