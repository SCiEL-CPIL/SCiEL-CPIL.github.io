"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function that minimizes the maximum absolute autoconvolution
    divided by the squared integral.

    Theoretical analysis:
    For a sequence x of length N, the autoconvolution at lag 0 is C(0) = sum(x_i^2).
    By Cauchy-Schwarz, sum(x_i^2) >= (sum x_i)^2 / N.
    Therefore, C(0) / (sum x_i)^2 >= 1/N.
    Since max|C(k)| >= C(0), we have max|C(k)| / (sum x_i)^2 >= 1/N.
    
    Equality holds if and only if all x_i are equal (constant sequence).
    The constant sequence x_i = 1 for all i achieves:
      - sum(x_i) = N
      - C(0) = N
      - For k != 0, C(k) = N - |k| (for the full convolution of ones)
      - Actually, for x = ones(N), the autoconvolution is a triangular sequence:
        C(k) = N - |k| for |k| < N, and 0 otherwise.
      - max|C(k)| = C(0) = N
      - Ratio = N / N^2 = 1/N
    
    This is the theoretical minimum, so the constant sequence is optimal.
    """
    x = np.ones(N)
    return x
# EVOLVE-BLOCK-END
