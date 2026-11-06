"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a correct Odd-Even Transposition Sort network for N=16.
    This network uses 120 comparators and is guaranteed to be correct.
    """
    network = []
    N = 16
    # Odd-Even Transposition Sort (Parallel Bubble Sort)
    for step in range(N):
        if step % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network
# EVOLVE-BLOCK-END
