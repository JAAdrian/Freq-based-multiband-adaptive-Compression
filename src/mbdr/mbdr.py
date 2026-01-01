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
    """Compress the input signal based on freq-based multiband adaptive compression.

    Args:
        signal: Single-channel input audio signal.
        prescriptive_gains: Tuple of 9 gains in dB for each half-octave band.
        compression_threshold: The compressor's threshold in dB .
                               Defaults to DEFAULT_COMPRESSION_THRESHOLD.
        compression_ratio: The compressor's ratio as an integer.
                           Defaults to DEFAULT_COMPRESSION_RATIO.
        knee_width: The compressor's knee width. Defaults to DEFAULT_KNEE_WIDTH.
        makeup_gain: The applied makeup gain after compression in dB.
                     Defaults to DEFAULT_MAKEUP_GAIN.
        attack_time: The compressor's attack time in seconds.
                     Defaults to DEFAULT_ATTACK_TIME_SEC.
        sample_rate: The corresponding sample rate in Hz.
                     Defaults to DEFAULT_SAMPLE_RATE.

    Returns:
        Compressed single-channel time-domain signal.
    """
    fb = fbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=sample_rate,
    )

    stft = fb.analysis(x=signal)
    signal_magnitudes = numpy.abs(stft)

    if DEBUG:
        from matplotlib import pyplot

        fig, ax = pyplot.subplots(figsize=(12, 12 / 1.618))
        pc = ax.pcolormesh(20 * numpy.log10(signal_magnitudes))
        fig.colorbar(pc, ax=ax)

        pyplot.show()

    signal_plus_gain = _add_audiogram_gain(
        magnitudes=signal_magnitudes,
        filterbank=fb,
        gains=prescriptive_gains,
        center_frequencies_hertz=CENTER_FREQUENCIES_HZ,  # type: ignore
        sample_rate=sample_rate,
    )
    compression_function = _get_compression_function(
        amplified_power_spectrum=signal_plus_gain,
        compression_threshold=compression_threshold,
        compression_ratio=compression_ratio,
        knee_width=knee_width,
    )
    smoothed_gain_function = _smooth_gains(
        compression_function=compression_function,
        signal_magnitudes=signal_magnitudes,
        hop_size=fb.hop_size,
        attack_time_sec=attack_time,
        sample_rate=sample_rate,
    )

    compressed_magnitudes = stft * 10 ** ((makeup_gain + smoothed_gain_function) / 20)
    compressed_signal = fb.synthesis(
        stft=compressed_magnitudes, original_signal_length=len(signal)
    )
    return compressed_signal


def _add_audiogram_gain(
    magnitudes: numpy.ndarray,
    filterbank: fbank.Filterbank,
    gains: tuple[int],
    center_frequencies_hertz: tuple[int],
    sample_rate: int,
) -> numpy.ndarray:
    """Add the prescriptive gains to the actual power spectrum.

    This is the data basis on which the compression function is applied.

    Args:
        magnitudes: The audio signal's time-frequency representation in magnitude
                    scaling.
        filterbank: The used filterbank object.
        gains: The prescriptive audiogram gains in dB.
        center_frequencies_hertz: The audiogram's center frequencies in Hz.
        sample_rate: The corresponding sample rate in Hz.

    Raises:
        ValueError: Raised when the number of gains does not match the numer of center
                    frequencies.

    Returns:
        The power spectrum after applied prescriptive gains.
    """
    if len(gains) != len(center_frequencies_hertz):
        raise ValueError(
            "The number of gain values must match the number of center frequencies"
        )

    amplified_spectrum = 20 * numpy.log10(magnitudes)

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

        amplified_spectrum[lower_bin:upper_bin, :] += gains[frequency_index]

    return amplified_spectrum


def _get_compression_function(
    amplified_power_spectrum: numpy.ndarray,
    compression_threshold: int,
    compression_ratio: int,
    knee_width: int,
) -> numpy.ndarray:
    """Compute the compression function based on the amplified power spectrum.

    Args:
        amplified_power_spectrum: The amplified power spectrum in simple V²-scaling.
        compression_threshold: The compressor's threshold in dB.
        compression_ratio: The compressor's ratio.
        knee_width: The compressor's kneewidth.

    Returns:
        Compression function with the same dimensions and shape as the input spectrum.
    """
    compression_function = amplified_power_spectrum

    f_two = numpy.where(
        2 * numpy.abs(amplified_power_spectrum - compression_threshold) <= knee_width
    )
    compression_function[f_two] += (
        (1 / compression_ratio - 1)
        * (amplified_power_spectrum[f_two] - compression_threshold + knee_width / 2)
        ** 2
        / (2 * knee_width)
    )

    f_three = numpy.where(
        2 * (amplified_power_spectrum - compression_threshold) > knee_width
    )
    compression_function[f_three] = (
        compression_threshold
        + (amplified_power_spectrum[f_three] - compression_threshold)
        / compression_ratio
    )

    return compression_function


def _smooth_gains(
    compression_function: numpy.ndarray,
    signal_magnitudes: numpy.ndarray,
    hop_size: int,
    attack_time_sec: float,
    sample_rate: int,
) -> numpy.ndarray:
    """Gain smoothing which applies an adaptive release time in t-f-domain.

    Args:
        compression_function: The compression function which is the amplification
                              matrix.
        signal_magnitudes: The input signal's magnitudes in plain STFT scaling.
        hop_size: The transform's hop size (frameshift) in samples.
        attack_time_sec: The desired attack time in seconds.
        sample_rate: The corresponding sample rate in hZ.

    Returns:
        The final gain matrix after compression and gain smoothing.
    """
    power_spectrum = 20 * numpy.log10(signal_magnitudes)

    # This is W_G in equation (5).
    compression_residual = compression_function - power_spectrum
    delayed_compression_residual = numpy.roll(compression_residual, shift=1, axis=1)
    delayed_compression_residual = delayed_compression_residual[:, :-1]

    # Compute spectral flux and smooth it based on a guessed time constant.
    flux = utils.spectral_flux(magnitudes=signal_magnitudes)

    frame_rate = sample_rate // hop_size
    smoother = utils.RecursiveSmoother(
        time_series=flux,
        time_constant=FLUX_SMOOTHING_CONSTANT_SEC,
        sample_rate=frame_rate,
    )
    smoothed_flux = numpy.fromiter(smoother, dtype=float)

    # Compute the adaptive release time from equation (6).
    adaptive_release_time = numpy.maximum(
        MINIMUM_RELEASE_TIME_SEC,
        MINIMUM_RELEASE_TIME_SEC / (smoothed_flux**FLUX_GAMMA + FLUX_EPSILON),
    )

    # Introduce an upper bound to not overshoot the MAXIMUM_RELEASE_TIME_SEC.
    adaptive_release_time = numpy.minimum(
        MAXIMUM_RELEASE_TIME_SEC, adaptive_release_time
    )
    alphas_release = 1 - numpy.exp(-1 / (adaptive_release_time * frame_rate))

    # Iterate over each time frame and smooth by applying attack and release times.
    smoothed_gains = list()
    alpha_attack = 1 - numpy.exp(-1 / (attack_time_sec * frame_rate))
    for index, (residual, alpha_release) in enumerate(
        zip(delayed_compression_residual.T, alphas_release, strict=True)
    ):
        compression_gain = compression_function[:, index]

        frame_gain = numpy.zeros_like(residual)
        residual_greater_than_compression = residual > compression_gain
        residual_less_than_compression = residual <= compression_gain

        if numpy.sum(residual_greater_than_compression):
            frame_gain[residual_greater_than_compression] = (
                alpha_attack * residual + (1 - alpha_attack) * compression_gain
            )
        if numpy.sum(residual_less_than_compression):
            frame_gain[residual_less_than_compression] = (
                alpha_release * residual + (1 - alpha_release) * compression_gain
            )

        smoothed_gains.append(frame_gain)

    # Add the last frame to counteract the loss of one frame due to spectral flux.
    smoothed_gains.append(smoothed_gains[-1])
    return numpy.stack(smoothed_gains, axis=1)
