"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Growth index approximation:
    # gamma ~= 0.55 - 0.05 * u
    # where u = 1 + w0 + wa*(1-a)
    # For LCDM (w=-1, u=0), gamma = 0.55.
    # For EdS-like (w=0, u=1), gamma = 0.50.
    c0 = 0.55
    c1 = 0.0
    c2 = 0.0
    c3 = -0.05
    c4 = 0.0
    c5 = 0.0
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
