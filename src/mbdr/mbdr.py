"""Implementation of Patel's and Panahi's MBDR.

Compressor Paramters:

- Threshold `T`
- Ratio `R`
- Knee Width `W`
- Attack Time `tau_A`
- Release Time `tau_R`
- Makeup Gain `M`
- Number of Bands
"""

import numpy

from mbdr import filterbank as fbank
from mbdr import utils

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_COMPRESSION_THRESHOLD = -30
DEFAULT_COMPRESSION_RATIO = 5
DEFAULT_KNEE_WIDTH = 4
DEFAULT_MAKEUP_GAIN = 0
DEFAULT_ATTACK_TIME_SEC = 20e-3
DEFAULT_RELEASE_TIME_SEC = 1e-1

CENTER_FREQUENCIES_HZ = (250, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 6_000, 8_000)

BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5

FLUX_SMOOTHING_CONSTANT_SEC = 100e-3
FLUX_EPSILON = 1e-16
FLUX_GAMMA = 4

MINIMUM_RELEASE_TIME_SEC = 0.2
MAXIMUM_RELEASE_TIME_SEC = 2

DEBUG = False


def compress_signal(
    signal: numpy.ndarray,
    prescriptive_gains: tuple[int],
    compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
    compression_ratio: int = DEFAULT_COMPRESSION_RATIO,
    knee_width: int = DEFAULT_KNEE_WIDTH,
    makeup_gain: int = DEFAULT_MAKEUP_GAIN,
    attack_time: float = DEFAULT_ATTACK_TIME_SEC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> numpy.ndarray:
    """Compress the input signal based on freq-based multiband adaptive compression."""
    fb = fbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=sample_rate,
    )

    stft = fb.analysis(x=signal)
    power_spectrum = 20 * numpy.log10(numpy.abs(stft))

    if DEBUG:
        from matplotlib import pyplot

        fig, ax = pyplot.subplots(figsize=(12, 12 / 1.618))
        pc = ax.pcolormesh(power_spectrum)
        fig.colorbar(pc, ax=ax)

        pyplot.show()

    signal_plus_gain = _add_gain(
        power_spectrum=power_spectrum,
        filterbank=fb,
        gains=prescriptive_gains,
        center_frequencies_hertz=CENTER_FREQUENCIES_HZ,  # type: ignore
        sample_rate=sample_rate,
    )
    compression_function = _get_compression_function(
        power_spectrum=signal_plus_gain,
        compression_threshold=compression_threshold,
        compression_ratio=compression_ratio,
        knee_width=knee_width,
    )
    smoothed_gain_function = _smooth_gains(
        signal_magnitudes=numpy.abs(stft),
        compression_function=compression_function,
        hop_size=fb.hop_size,
        attack_time=attack_time,
        sample_rate=sample_rate,
    )

    compressed_magnitudes = stft * 10 ** ((makeup_gain + smoothed_gain_function) / 20)
    compressed_signal = fb.synthesis(
        spectrum=compressed_magnitudes, original_signal_length=len(signal)
    )
    return compressed_signal


def _add_gain(
    power_spectrum: numpy.ndarray,
    filterbank: fbank.Filterbank,
    gains: tuple[int],
    center_frequencies_hertz: tuple[int],
    sample_rate: int,
) -> numpy.ndarray:
    if len(gains) != len(center_frequencies_hertz):
        raise ValueError(
            "The number of gain values must match the number of center frequencies"
        )

    output_spectrum = power_spectrum

    for frequency_index, center_frequency in enumerate(center_frequencies_hertz):
        lower_bin = fbank.get_bin_index(
            frequency=fbank.get_lower_edge_frequency(center_frequency=center_frequency),
            fft_size=filterbank.fft_size,
            sample_rate=sample_rate,
        )
        upper_bin = fbank.get_bin_index(
            frequency=fbank.get_upper_edge_frequency(center_frequency=center_frequency),
            fft_size=filterbank.fft_size,
            sample_rate=sample_rate,
        )

        output_spectrum[lower_bin:upper_bin, :] += gains[frequency_index]

    return output_spectrum


def _get_compression_function(
    power_spectrum: numpy.ndarray,
    compression_threshold: int,
    compression_ratio: int,
    knee_width: int,
) -> numpy.ndarray:
    compression_function = power_spectrum

    f_two = numpy.where(
        2 * numpy.abs(power_spectrum - compression_threshold) <= knee_width
    )
    compression_function[f_two] += (
        (1 / compression_ratio - 1)
        * (power_spectrum[f_two] - compression_threshold + knee_width / 2) ** 2
        / (2 * knee_width)
    )

    f_three = numpy.where(2 * (power_spectrum - compression_threshold) > knee_width)
    compression_function[f_three] = (
        compression_threshold
        + (power_spectrum[f_three] - compression_threshold) / compression_ratio
    )

    return compression_function


def _smooth_gains(
    signal_magnitudes: numpy.ndarray,
    compression_function: numpy.ndarray,
    hop_size: int,
    attack_time: float,
    sample_rate: int,
) -> numpy.ndarray:
    power_spectrum = 20 * numpy.log10(signal_magnitudes)
    compression_residual = compression_function - power_spectrum

    flux = utils.spectral_flux(magnitudes=signal_magnitudes)

    frame_rate = sample_rate // hop_size
    smoother = utils.RecursiveSmoother(
        time_series=flux,
        time_constant=FLUX_SMOOTHING_CONSTANT_SEC,
        sample_rate=frame_rate,
    )
    smoothed_flux = numpy.fromiter(smoother, dtype=float)
    adaptive_release_time = numpy.maximum(
        MINIMUM_RELEASE_TIME_SEC,
        MINIMUM_RELEASE_TIME_SEC / (smoothed_flux**FLUX_GAMMA + FLUX_EPSILON),
    )
    adaptive_release_time = numpy.minimum(
        MAXIMUM_RELEASE_TIME_SEC, adaptive_release_time
    )
    alphas_release = 1 - numpy.exp(-1 / (adaptive_release_time * frame_rate))

    smoothed_gains = list()
    alpha_attack = 1 - numpy.exp(-1 / (attack_time * frame_rate))
    for index, (residual, alpha_release) in enumerate(
        zip(compression_residual[:, 1:].T, alphas_release)
    ):
        compression_value = compression_function[:, index]

        new_gain = numpy.zeros_like(residual)
        new_gain[residual > compression_value] = (
            alpha_attack * residual + (1 - alpha_attack) * compression_value
        )
        new_gain[residual >= compression_value] = (
            alpha_release * residual + (1 - alpha_release) * compression_value
        )

        smoothed_gains.append(new_gain)

    smoothed_gains.append(smoothed_gains[-1])
    return numpy.stack(smoothed_gains, axis=1)
