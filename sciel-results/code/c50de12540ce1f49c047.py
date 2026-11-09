"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0:
            candidates.append(i)
    return candidates
# EVOLVE-BLOCK-END
