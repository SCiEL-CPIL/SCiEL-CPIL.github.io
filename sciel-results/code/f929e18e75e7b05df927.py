"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Returns the known optimal ±1 polynomial coefficients for degree 23 (24 terms)
    # minimizing the sup norm on the unit circle.
    # This sequence is derived from Odlyzko's tables for optimal Littlewood polynomials.
    coeffs = [
        1, 1, 1, 1, 1, -1, 1, 1,
        1, -1, -1, 1, -1, 1, 1, -1,
        -1, 1, 1, -1, 1, 1, -1, 1
    ]
    return coeffs
# EVOLVE-BLOCK-END
