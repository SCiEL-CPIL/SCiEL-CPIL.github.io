N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a single spike to minimize the autoconvolution peak.
    # This maximizes the ratio as the denominator is minimized.
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
