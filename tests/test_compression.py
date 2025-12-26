"""Test the compression algorithm."""

import pytest
import soundfile

from mbdr import mbdr

TEST_SIGNAL_FILEPATH = r"tests/audio/61-70968-0049.flac"


@pytest.fixture
def audio_signal():
    """Read test audio signal."""
    return soundfile.read(TEST_SIGNAL_FILEPATH)


def test_compression(audio_signal):
    """Test whether the multiband compression returns a signal."""
    signal, sample_rate = audio_signal
    gains = 9 * (0,)

    compression_threshold = -60
    compression_ratio = 2
    knee_width = 5
    makeup_gain = 5

    compressed_signal = mbdr.compress_signal(
        signal=signal,
        prescriptive_gains=gains,  # type: ignore
        compression_threshold=compression_threshold,
        compression_ratio=compression_ratio,
        knee_width=knee_width,
        makeup_gain=makeup_gain,
        sample_rate=sample_rate,
    )

    assert len(compressed_signal) == len(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
