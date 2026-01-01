"""Filterbank implementation and tools."""

import numpy
from scipy import signal
from scipy.signal import windows

WINDOW_FUNCTION = "hamming"
DEFAULT_BLOCK_SIZE_SEC = 32e-3
DEFAULT_OVERLAP_RATIO = 0.5


def compute_stft(
    x: numpy.ndarray,
    window: numpy.ndarray,
    hop_size: int,
    fft_size: int,
    sample_rate: int,
) -> tuple[numpy.ndarray, signal.ShortTimeFFT]:
    """Compute the STFT for a given single-channel audio signal.

    This implementation uses the `ShortTimeFFT` class of the scipy stack.

    Args:
        x: Single-channel audio signal
        window: Desired window function as a numpy array
        hop_size: The frame shift or hop size in samples. This states the shift between
                  overlapping frames. Usually around 50% of the block size in samples.
        fft_size: The size of the DFT in bins. Usually a next power of 2 wrt the block
                  size in samples.
        sample_rate: The signal's sample rate

    Returns:
        Tuple of STFT matrix and corresponding scipy object.
    """
    transform = signal.ShortTimeFFT(
        win=window,
        hop=hop_size,
        fs=sample_rate,
        fft_mode="onesided",
        mfft=fft_size,
        scale_to="magnitude",
    )
    return transform.stft(x), transform


def get_fft_size(block_size: int) -> int:
    """Return the DFT size in bins as the next power of 2 wrt the block size."""
    return int(2 ** (numpy.ceil(numpy.log2(block_size))))


class Filterbank:
    """A filterbank class which uses the `ShortTimeFFT` implementation of scipy.

    This class implements an analysis and synthesis step to get to and from
    time-frequency representation.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size_sec: float = DEFAULT_BLOCK_SIZE_SEC,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ):
        """Instantiate a filterbank object.

        Args:
            sample_rate: The single-channel audio's sample rate in Hz.
            block_size_sec: The desired block size in seconds.
                            Defaults to DEFAULT_BLOCK_SIZE_SEC.
            overlap_ratio: The desired overlap as a ratio between 0 and 1.
                           Defaults to DEFAULT_OVERLAP_RATIO.
        """
        self._transform: signal.ShortTimeFFT = None  # type: ignore

        self._sample_rate = sample_rate
        self._block_size = block_size = round(block_size_sec * sample_rate)
        self._overlap = round(block_size * overlap_ratio)
        self._hop_size = block_size - self._overlap
        self._fft_size = get_fft_size(self._block_size)

    def analysis(self, x: numpy.ndarray):
        """Return the STFT matrix by applying the analysis filterbank.

        Args:
            x: The single-channel audio signal.

        Returns:
            The STFT K-by-N matrix with K being the size of a single-sided DFT and N the
            number of time frames.
        """
        window = getattr(windows, WINDOW_FUNCTION)(self._block_size, sym=False)

        stft, self._transform = compute_stft(
            x=x,
            window=window,
            hop_size=self._hop_size,
            fft_size=self._fft_size,
            sample_rate=self._sample_rate,
        )
        return stft

    def synthesis(self, stft: numpy.ndarray, original_signal_length: int):
        """Return the time-domain signal by applying the synthesis filterbank.

        Args:
            stft: The signal's complex Time-frequency representation.
            original_signal_length: The input signal's original length in samples so
                                    that the output signal can be cropped to the same
                                    length.

        Returns:
            _description_
        """
        return self._transform.istft(S=stft)[:original_signal_length]

    @property
    def fft_size(self) -> int:
        """Get the applied DFT size of the current filterbank object."""
        return self._fft_size

    @property
    def hop_size(self) -> int:
        """Get the applied hop size (frame shift) of the current filterbank object."""
        return self._hop_size


def get_lower_edge_frequency(center_frequency: numpy.ndarray | int) -> numpy.ndarray:
    """Get lower edge frequency of a half-octave filter based on center frequencies."""
    return numpy.round(center_frequency / 2**0.25)


def get_upper_edge_frequency(center_frequency: numpy.ndarray | int) -> numpy.ndarray:
    """Get upper edge frequency of a half-octave filter based on center frequencies."""
    return numpy.round(center_frequency * 2**0.25)


def get_bin_index(
    frequency: float | numpy.ndarray, fft_size: int, sample_rate: int
) -> int:
    """Get the corresponding bin index for a frequency given DFT-parameters."""
    bin_resolution = sample_rate / fft_size
    return round(frequency / bin_resolution)  # type: ignore
