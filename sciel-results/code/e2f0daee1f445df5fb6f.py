"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Superincreasing sequence ensures all subset sums are unique,
    # which implies sums of different cardinalities are disjoint.
    # Powers of 2 up to 256.
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

# EVOLVE-BLOCK-END
