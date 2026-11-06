"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Cycle + halving map: letter 0 is a cycle, letter 1 halves (floor(i/2)).
    # This creates a slower collapse than a constant map.
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = [i // 2 for i in range(N)]
    return [letter0, letter1]
# EVOLVE-BLOCK-END
