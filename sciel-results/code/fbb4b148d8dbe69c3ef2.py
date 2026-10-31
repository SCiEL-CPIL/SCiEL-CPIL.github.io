N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a Sidon set to minimize the peak of the autoconvolution
    # while maintaining a moderate L2 norm relative to L1 norm.
    # The set {1, 2, 4, 8, 16, 32, 49} has distinct pairwise sums.
    indices = [1, 2, 4, 8, 16, 32, 49]
    h = [0.0] * N
    for i in indices:
        h[i] = 1.0
    return h
# EVOLVE-BLOCK-END
