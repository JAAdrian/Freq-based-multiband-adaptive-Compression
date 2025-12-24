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
    gains = 9 * (0)

    compressed_signal = mbdr.compress_signal(
        signal=signal, prescriptive_gains=gains, sample_rate=sample_rate
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
