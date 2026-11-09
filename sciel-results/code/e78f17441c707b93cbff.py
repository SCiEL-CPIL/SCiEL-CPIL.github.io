"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    return bin((1 << N) - 1)[2:]
# EVOLVE-BLOCK-END
