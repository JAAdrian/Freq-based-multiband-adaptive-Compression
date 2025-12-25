"""Oridnary recursive smoothing functions."""

import numpy


def spectral_flux(magnitudes: numpy.ndarray) -> numpy.ndarray:
    spectral_difference = magnitudes[:, 1:] - magnitudes[:, :-1]
    return numpy.mean(spectral_difference**2, axis=0)


def RecursiveSmoother(
    time_series: numpy.ndarray,
    time_constant: float,
    sample_rate: int,
    initial_value: float = 0.0,
):
    previous_value = initial_value
    alpha = 1 - numpy.exp(-1 / (time_constant * sample_rate))

    for sample in time_series:
        smoothed_sample = alpha * sample + (1 - alpha) * previous_value
        previous_value = smoothed_sample

        yield smoothed_sample
