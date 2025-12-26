import numpy
from scipy import signal
from scipy.signal import windows

WINDOW_FUNCTION = "hamming"
DEFAULT_BLOCK_SIZE_SEC = 32e-3
DEFAULT_OVERLAP_RATIO = 0.5


def compute_stft(
    x: numpy.ndarray,
    window: numpy.ndarray,
    sample_rate: int,
    hop_size: int,
    fft_size: int,
) -> tuple[numpy.ndarray, signal.ShortTimeFFT]:
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
    return int(2 ** (numpy.ceil(numpy.log2(block_size))))


class Filterbank:
    def __init__(
        self,
        sample_rate: int,
        block_size_sec: float = DEFAULT_BLOCK_SIZE_SEC,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ):
        self._transform: signal.ShortTimeFFT = None  # type: ignore

        self._sample_rate = sample_rate
        self._block_size = block_size = round(block_size_sec * sample_rate)
        self._overlap = round(block_size * overlap_ratio)
        self._hop_size = block_size - self._overlap
        self._fft_size = get_fft_size(self._block_size)

    def analysis(self, x: numpy.ndarray):
        window = getattr(windows, WINDOW_FUNCTION)(self._block_size, sym=False)

        stft, self._transform = compute_stft(
            x=x,
            window=window,
            sample_rate=self._sample_rate,
            hop_size=self._hop_size,
            fft_size=self._fft_size,
        )
        return stft

    def synthesis(self, spectrum: numpy.ndarray, original_signal_length: int):
        return self._transform.istft(S=spectrum)[:original_signal_length]

    @property
    def fft_size(self) -> int:
        return self._fft_size


def get_lower_edge_frequency(center_frequency: numpy.ndarray | int) -> numpy.ndarray:
    return numpy.round(center_frequency / 2**0.25)


def get_upper_edge_frequency(center_frequency: numpy.ndarray | int) -> numpy.ndarray:
    return numpy.round(center_frequency * 2**0.25)


def get_bin_index(
    frequency: float | numpy.ndarray, fft_size: int, sample_rate: int
) -> int:
    bin_resolution = sample_rate / fft_size
    return round(frequency / bin_resolution)
