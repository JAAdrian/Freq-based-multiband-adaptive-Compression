import numpy
from scipy import signal
from scipy.signal import windows

WINDOW_FUNCTION = "hamming"
DEFAULT_BLOCK_SIZE_SEC = 32e-3
DEFAULT_OVERLAP_RATIO = 0.5


class Filterbank:
    def __init__(self):
        self._transform: signal.ShortTimeFFT = None  # type: ignore

    def analysis(
        self,
        x: numpy.ndarray,
        sample_rate: int,
        block_size_sec: float = DEFAULT_BLOCK_SIZE_SEC,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ):
        block_size = round(block_size_sec * sample_rate)
        overlap = round(block_size * overlap_ratio)

        hop_size = block_size - overlap
        window = getattr(windows, WINDOW_FUNCTION)(block_size, sym=False)
        fft_size = int(2 ** (numpy.ceil(numpy.log2(block_size))))

        self._transform = signal.ShortTimeFFT(
            win=window,
            hop=hop_size,
            fs=sample_rate,
            fft_mode="onesided",
            mfft=fft_size,
            scale_to="magnitude",
        )
        return self._transform.stft(x)

    def synthesis(self, spectrum: numpy.ndarray):
        return self._transform.istft(S=spectrum)
