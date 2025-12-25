"""Unit tests for the recursive smoother."""

import pytest
import numpy
from numpy import testing

from mbdr import utils

LEN_NOISE_SEC = 1
SAMPLE_RATE = 16_000

TIME_CONSTANT = 0.5


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


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
