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

SAMPLE_RATE = 16_000
CENTER_FREQUENCIES_HZ = (250, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 6_000, 8_000)

BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5

DEBUG = True


def compress_signal(
    signal: numpy.ndarray,
    prescriptive_gains: tuple[int],
    compression_threshold: int,
    compression_ratio: int,
    knee_width: int,
    sample_rate: int = SAMPLE_RATE,
) -> numpy.ndarray:
    """Compress the input signal based on freq-based multiband adaptive compression."""
    fb = fbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=sample_rate,
    )

    stft = fb.analysis(x=signal)
    original_phase = numpy.angle(stft)
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
    smoothed_power_spectrum = _smooth_gains(compression_function=compression_function)

    smoothed_magnitudes = 10 ** (smoothed_power_spectrum / 20)
    compressed_signal = fb.synthesis(
        spectrum=smoothed_magnitudes * numpy.exp(1j * original_phase),
        original_signal_length=len(signal),
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


def _smooth_gains(compression_function: numpy.ndarray) -> numpy.ndarray:
    pass
