"""Test the compression algorithm."""

import numpy
import pytest
import soundcard
import soundfile
from matplotlib import pyplot
from numpy import testing
from scipy import signal

from mbdr import mbdr

TEST_SIGNAL_FILEPATH = r"tests/audio/61-70968-0049.flac"


def _rms(audio: numpy.ndarray) -> float:
    return numpy.sqrt(numpy.mean(audio**2))


def _compute_welch(
    audio: numpy.ndarray, sample_rate: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    block_size_sec = 100e-3
    overlap_ratio = 0.5

    block_size = round(block_size_sec * sample_rate)
    overlap = round(block_size * overlap_ratio)
    fft_size = int(2 ** (numpy.ceil(numpy.log2(block_size))))
    win = "hann"

    frequency, psd = signal.welch(
        x=audio,
        fs=sample_rate,
        window=win,
        nperseg=block_size,
        noverlap=overlap,
        nfft=fft_size,
        scaling="spectrum",
    )

    return frequency, psd


def _plot_welch(
    original_signal: numpy.ndarray, compressed_signal: numpy.ndarray, sample_rate: int
):
    frequency, psd_original = _compute_welch(original_signal, sample_rate=sample_rate)
    _, psd_compressed = _compute_welch(compressed_signal, sample_rate=sample_rate)

    _, ax = pyplot.subplots(figsize=(12, 12 / 1.618))
    ax.semilogx(
        frequency, 10 * numpy.log10(psd_original), label="Original", linewidth=2
    )
    ax.semilogx(
        frequency,
        10 * numpy.log10(psd_compressed),
        label="Amplified and Compressed",
        linewidth=2,
    )
    ax.legend()
    ax.grid()
    pyplot.show()


def _plot_spectrogram(
    original_signal: numpy.ndarray, compressed_signal: numpy.ndarray, sample_rate: int
):
    time, frequency, stft_original = _compute_stft(
        original_signal, sample_rate=sample_rate
    )
    _, _, stft_compressed = _compute_stft(compressed_signal, sample_rate=sample_rate)

    _, ax = pyplot.subplots(nrows=2, figsize=(12, 12 / 1.618), sharex=True, sharey=True)
    mesh = ax[0].pcolormesh(time, frequency, 20 * numpy.log10(stft_original))
    ax[0].set_title("Original")
    ax[0].set_ylabel("Frequency / Hz")
    pyplot.colorbar(mesh)

    mesh = ax[1].pcolormesh(time, frequency, 20 * numpy.log10(stft_compressed))
    ax[1].set_title("Amplified and Compressed")
    ax[1].set_ylabel("Frequency / Hz")
    ax[1].set_xlabel("Time / s")
    pyplot.colorbar(mesh)

    pyplot.show()


def _compute_stft(audio: numpy.ndarray, sample_rate: int):
    block_size_sec = 32e-3
    overlap_ratio = 0.5

    block_size = round(block_size_sec * sample_rate)
    overlap = round(block_size * overlap_ratio)
    hop_size = block_size - overlap
    fft_size = int(2 ** (numpy.ceil(numpy.log2(block_size))))
    win = signal.windows.get_window("hann", block_size, fftbins=True)

    stft = signal.ShortTimeFFT(win=win, hop=hop_size, fs=sample_rate, mfft=fft_size)
    spec = stft.stft(x=audio)

    time = stft.t(len(audio), k_offset=block_size // 2)
    frequency = stft.f

    return time, frequency, numpy.abs(spec)


@pytest.fixture
def signal_and_samplerate():
    """Read test audio signal."""
    return soundfile.read(TEST_SIGNAL_FILEPATH)


def test_compression(signal_and_samplerate):
    """Test whether the multiband compression returns a signal."""
    original_signal, sample_rate = signal_and_samplerate
    gains = (0, 0, 10, 15, 2, 5, 0, 0, 30)

    compression_threshold = -50
    compression_ratio = 2
    knee_width = 5
    makeup_gain = 20

    compressed_signal = mbdr.compress_signal(
        signal=original_signal,
        prescriptive_gains=gains,  # type: ignore
        compression_threshold=compression_threshold,
        compression_ratio=compression_ratio,
        knee_width=knee_width,
        makeup_gain=makeup_gain,
        attack_time=20e-3,
        sample_rate=sample_rate,
    )

    _, ax = pyplot.subplots(figsize=(12, 12 / 1.618))
    ax.plot(original_signal, label="Original")
    ax.plot(compressed_signal, label="Compressed")
    ax.legend()
    pyplot.show()

    assert len(compressed_signal) == len(original_signal)
    testing.assert_almost_equal(
        _rms(compressed_signal), _rms(original_signal), decimal=1
    )

    player = soundcard.default_speaker()
    player.play(data=original_signal, samplerate=sample_rate)
    player.play(data=compressed_signal, samplerate=sample_rate)

    _plot_welch(original_signal, compressed_signal, sample_rate)
    _plot_spectrogram(original_signal, compressed_signal, sample_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])
