"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a signed step function using the Thue-Morse sequence.
    The Thue-Morse sequence is a deterministic, low-autocorrelation sequence
    that breaks the symmetry of the all-ones seed and reduces the
    autoconvolution upper bound by minimizing off-peak spectral components.
    """
    tm = np.zeros(N, dtype=np.float64)
    tm[0] = 1.0
    for i in range(1, N):
        if i % 2 == 0:
            tm[i] = tm[i // 2]
        else:
            tm[i] = -tm[i // 2]
    return tm
# EVOLVE-BLOCK-END
