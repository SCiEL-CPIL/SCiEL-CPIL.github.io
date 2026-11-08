"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    # Use a low-frequency cosine wave to reduce the maximum overlap.
    # h_i = 0.5 + 0.5 * cos(2*pi*i/N)
    # This keeps h_i in [0, 1] and sum(h) = N/2.
    # It reduces O_0 significantly and keeps other O_k below the constant sequence's max.
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    return h

# EVOLVE-BLOCK-END
