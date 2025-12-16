import numpy
from numpy import testing
import pytest

from mbdr.filterbank import Filterbank

SAMPLE_RATE = 16_000
BLOCK_SIZE_SEC = 32e-3
OVERLAP_RATIO = 0.5


def test_analysis():
    impulse = numpy.array(numpy.eye(N=512, M=1)).ravel()

    filterbank = Filterbank()
    spec = filterbank.analysis(
        x=impulse,
        block_size_sec=BLOCK_SIZE_SEC,
        overlap_ratio=OVERLAP_RATIO,
        sample_rate=SAMPLE_RATE,
    )
    magnitudes = numpy.abs(spec)

    testing.assert_almost_equal(magnitudes.sum(axis=0).sum(), 1, decimal=2)


def test_synthesis():
    noise = numpy.random.randn(1 * SAMPLE_RATE)

    filterbank = Filterbank()
    noise_synthesized = filterbank.synthesis(
        filterbank.analysis(
            x=noise,
            block_size_sec=BLOCK_SIZE_SEC,
            overlap_ratio=OVERLAP_RATIO,
            sample_rate=SAMPLE_RATE,
        )
    )

    testing.assert_almost_equal(noise_synthesized, noise)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
