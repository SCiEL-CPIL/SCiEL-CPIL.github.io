"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a single spike with very small amplitude
    # For h = [ε, 0, ..., 0]: ratio = 1/ε
    # Using ε = 1e-10 for numerical stability
    result = [0.0] * N
    result[0] = 1e-10
    return result
# EVOLVE-BLOCK-END
