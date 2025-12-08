"""Implementation of Patel's and Panahi's MBDR.

Compressor Paramters:

- Threshold `T`
- Ratio `R`
- Knee Width `W`
- Attack Time `tau_A`
- Release Time `tau_R`
- Makeup Gain `M`
- Number of Bands
"""

import numpy

SAMPLE_RATE = 16_000
CENTER_FREQUENCIES_HZ = (250, 500, 1000, 1500, 2_000, 3_000, 4_000, 6_000, 8_000)


def compute_STFT() -> numpy.ndarray:
    pass


def get_band_bin_edges() -> numpy.ndarray:
    pass


if __name__ == "__main__":
    print(CENTER_FREQUENCIES_HZ)
