"""Test suite for the filterbank module."""

import numpy
import pytest
from numpy import testing

from mbdr import filterbank

SAMPLE_RATE = 16_000
BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5


def test_analysis():
    """Test wether the analysis works."""
    impulse = numpy.array(numpy.eye(N=512, M=1)).ravel()

    fb = filterbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=SAMPLE_RATE,
    )
    spec = fb.analysis(x=impulse)
    magnitudes = numpy.abs(spec)

    testing.assert_almost_equal(magnitudes.sum(axis=0).sum(), 1, decimal=2)


def test_synthesis():
    """Test wether the synthesis works."""
    noise = numpy.random.randn(1 * SAMPLE_RATE)

    fb = filterbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=SAMPLE_RATE,
    )
    noise_synthesized = fb.synthesis(
        fb.analysis(x=noise),
        original_signal_length=len(noise),
    )

    testing.assert_almost_equal(noise_synthesized, noise)


def test_edge_frequencies():
    """Test the edge frequency computation."""
    center_frequencies = numpy.array([500, 1000, 1500])
    expected_lower_frequencies = numpy.array([420, 841, 1261])
    expected_upper_frequencies = numpy.array([595, 1189, 1784])

    testing.assert_almost_equal(
        filterbank.get_lower_edge_frequency(center_frequency=center_frequencies),
        expected_lower_frequencies,
    )

    testing.assert_almost_equal(
        filterbank.get_upper_edge_frequency(center_frequency=center_frequencies),
        expected_upper_frequencies,
    )


def test_frequency_to_bin():
    """Test the frequency to bin conversion."""
    sample_rate = 16_000
    fft_size = 2048

    assert (
        filterbank.get_bin_index(
            frequency=0, fft_size=fft_size, sample_rate=sample_rate
        )
        == 0
    )

    assert (
        filterbank.get_bin_index(
            frequency=8000, fft_size=fft_size, sample_rate=sample_rate
        )
        == 1024
    )


def test_fft_size():
    """Test whether the correct fft size is computed."""
    fb = filterbank.Filterbank(
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=SAMPLE_RATE,
    )

    assert fb.fft_size == 512


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
