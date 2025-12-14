import numpy
from numpy import testing
import pytest

from mbdr.filterbank import Filterbank


def test_analysis():
    impulse = numpy.array(numpy.eye(N=512, M=1)).ravel()

    block_size_sec = 32e-3
    overlap_ratio = 0.5
    sample_rate = 16_000

    filterbank = Filterbank()
    spec = filterbank.analysis(
        x=impulse,
        block_size_sec=block_size_sec,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
    )
    magnitudes = numpy.abs(spec)

    testing.assert_almost_equal(magnitudes.sum(axis=0).sum(), 1, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
