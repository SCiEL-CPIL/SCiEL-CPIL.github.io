"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # A delta spike maximizes the ratio of L2^2 to (L1 * ConvPeak)
    # assuming standard discrete convolution and no external normalization
    # that would flatten the peakiness advantage.
    # If normalization to L1=1 is applied, the spike is still optimal
    # (or tied with box, but spike is safer against L2 normalization).
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
