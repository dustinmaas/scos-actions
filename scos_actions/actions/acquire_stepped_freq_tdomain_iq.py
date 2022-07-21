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
from scos_actions.actions.acquire_single_freq_tdomain_iq import (
    SingleFrequencyTimeDomainIqAcquisition,
)
from scos_actions.utils import get_parameter
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.metadata.sigmf_builder import Domain, MeasurementType
from scos_actions.hardware import gps as mock_gps

logger = logging.getLogger(__name__)

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"


class SteppedFrequencyTimeDomainIqAcquisition(SingleFrequencyTimeDomainIqAcquisition):
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

    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters=parameters, sigan=sigan, gps=gps)
        self.sorted_measurement_parameters = []
        num_center_frequencies = len(parameters[FREQUENCY])

        # convert dictionary of lists from yaml file to list of dictionaries
        parameters.pop('name')
        self.sorted_measurement_parameters = [dict(zip(parameters, v)) for v in zip(*parameters.values())]
        self.sorted_measurement_parameters.sort(key=lambda params: params[FREQUENCY])

        self.sigan = sigan  # make instance variable to allow mocking
        self.num_center_frequencies = num_center_frequencies

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()

        for recording_id, measurement_params in enumerate(
            self.sorted_measurement_parameters, start=1
        ):
            start_time = utils.get_datetime_str_now()
            self.configure(measurement_params)
            duration_ms = get_parameter(DURATION_MS, measurement_params)
            nskip = get_parameter(NUM_SKIP, measurement_params)
            sample_rate = self.sigan.sample_rate
            num_samples = int(sample_rate * duration_ms * 1e-3)
            measurement_result = super().acquire_data(num_samples, nskip)
            measurement_result.update(measurement_params)
            end_time = utils.get_datetime_str_now()
            measurement_result['start_time'] = start_time
            measurement_result['end_time'] = end_time
            measurement_result['domain'] = Domain.TIME.value
            measurement_result['measurement_type'] = MeasurementType.SINGLE_FREQUENCY.value
            measurement_result['task_id'] = task_id
            measurement_result['description'] = self.description
            measurement_result['name'] = self.name
            measurement_result['sigan_cal'] = self.sigan.sigan_calibration_data
            measurement_result['sensor_cal'] = self.sigan.sensor_calibration_data
            sigmf_builder = self.get_sigmf_builder(measurement_result)
            self.create_metadata(
                sigmf_builder, schedule_entry_json, measurement_result, recording_id
            )
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=measurement_result["data"],
                metadata=sigmf_builder.metadata,
            )

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""

        acquisition_plan = ""
        used_keys = [FREQUENCY, DURATION_MS, "name"]
        acq_plan_template = "The signal analyzer is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        acq_plan_template += "{parameters}"
        acq_plan_template += "Then, acquire samples for {duration_ms} ms.\n"

        for measurement_params in self.sorted_measurement_parameters:
            parameters = ""
            for name, value in measurement_params.items():
                if name not in used_keys:
                    parameters += f"{name} = {value}\n"
            acquisition_plan += acq_plan_template.format(
                **{
                    "center_frequency": measurement_params[FREQUENCY] / 1e6,
                    "parameters": parameters,
                    "duration_ms": measurement_params[DURATION_MS],
                }
            )

        durations = [v[DURATION_MS] for v in self.sorted_measurement_parameters]
        min_duration_ms = np.sum(durations)

        defs = {
            "name": self.name,
            "num_center_frequencies": self.num_center_frequencies,
            "center_frequencies": ", ".join(
                [
                    "{:.2f} MHz".format(param[FREQUENCY] / 1e6)
                    for param in self.sorted_measurement_parameters
                ]
            ),
            "acquisition_plan": acquisition_plan,
            "min_duration_ms": min_duration_ms,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
