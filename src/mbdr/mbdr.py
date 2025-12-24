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

from mbdr import filterbank

SAMPLE_RATE = 16_000
CENTER_FREQUENCIES_HZ = (250, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 6_000, 8_000)

BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5

DEBUG = True


def compress_signal(
    signal: numpy.ndarray,
    prescriptive_gains: numpy.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> numpy.ndarray:
    """Compress the input signal based on freq-based multiband adaptive compression."""
    fb = filterbank.Filterbank()

    stft = fb.analysis(
        x=signal,
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=sample_rate,
    )
    power_spectrum = 20 * numpy.log10(numpy.abs(stft))

    if DEBUG:
        from matplotlib import pyplot

        _, ax = pyplot.subplots(figsize=(12, 12 / 1.618))
        ax.pcolormesh(power_spectrum)
        pyplot.show()

    compressed_stft = _apply_compression(stft=stft, gains=prescriptive_gains)
    smoothed_stft = _smooth_gains(stft=compressed_stft)

    compressed_signal = fb.synthesis(smoothed_stft)
    return compressed_signal


def get_band_bin_edges() -> numpy.ndarray:
    """Return the bin indices for given center frequencies."""
    pass


def _apply_compression(stft: numpy.ndarray, gains: tuple):
    pass


def _smooth_gains(stft: numpy.ndarray):
    pass
