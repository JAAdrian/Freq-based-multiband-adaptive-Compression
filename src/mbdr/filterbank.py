from scipy import signal
from scipy.signal import windows
import numpy

WINDOW = "hann"


class Filterbank:
    def __init__(self):
        self._transform: signal.ShortTimeFFT = None # type: ignore

    def analysis(
        self,
        x: numpy.ndarray,
        block_size_sec: float,
        overlap_ratio: float,
        sample_rate: int,
    ):
        block_size = round(block_size_sec * sample_rate)
        overlap = round(block_size * overlap_ratio)

        hop_size = block_size - overlap
        window = windows.hann(block_size, sym=False)
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
