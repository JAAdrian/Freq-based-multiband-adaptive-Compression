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

    signal_plus_gain = _add_gain(
        power_spectrum=power_spectrum,
        filterbank=fb,
        gains=prescriptive_gains,
        center_frequencies_hertz=CENTER_FREQUENCIES_HZ,  # type: ignore
        sample_rate=sample_rate,
    )
    compressed_power_spectrum = _apply_compression(stft=signal_plus_gain)
    smoothed_power_spectrum = _smooth_gains(stft=compressed_power_spectrum)

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


def _apply_compression(stft: numpy.ndarray) -> numpy.ndarray:
    pass


def _smooth_gains(stft: numpy.ndarray) -> numpy.ndarray:
    pass
