"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    h = [0.0] * N
    # Sidon set in [0, 49]: {1, 2, 5, 10, 20, 40}
    # Pair sums are distinct, minimizing the peak of the autoconvolution
    # relative to the L2 norm for a given L1 norm.
    indices = [1, 2, 5, 10, 20, 40]
    for i in indices:
        if i < N:
            h[i] = 1.0
    return h
# EVOLVE-BLOCK-END
