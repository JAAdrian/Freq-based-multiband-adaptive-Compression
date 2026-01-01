"""Unit tests for the recursive smoother."""

import numpy
import pytest
from numpy import testing
from scipy import signal

from mbdr import filterbank, utils

LEN_NOISE_SEC = 1
SAMPLE_RATE = 16_000

TIME_CONSTANT = 0.5

BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5


def _get_stft(time_domain_signal) -> numpy.ndarray:
    block_size = round(BLOCK_SIZE_SEC * SAMPLE_RATE)
    overlap = round(block_size * OVERLAP_RATIO)
    hop_size = block_size - overlap
    fft_size = filterbank.get_fft_size(block_size)

    window = getattr(signal.windows, filterbank.WINDOW_FUNCTION)(block_size, sym=False)

    stft, _ = filterbank.compute_stft(
        x=time_domain_signal,
        window=window,
        hop_size=hop_size,
        fft_size=fft_size,
        sample_rate=SAMPLE_RATE,
    )
    return stft


@pytest.fixture
def noise():
    len_noise = round(SAMPLE_RATE * LEN_NOISE_SEC)
    return numpy.random.randn(len_noise)


def test_smoother(noise):
    gain_smoother = utils.RecursiveSmoother(
        time_series=noise, time_constant=TIME_CONSTANT, sample_rate=SAMPLE_RATE
    )

    result = numpy.fromiter(gain_smoother, dtype=float)
    testing.assert_almost_equal(result[-1], 0, decimal=2)


def test_spectral_flux(noise):
    magnitudes = numpy.abs(_get_stft(noise))

    flux = utils.spectral_flux(magnitudes)
    mean_flux = numpy.mean(flux)

    testing.assert_almost_equal(mean_flux, 0, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
