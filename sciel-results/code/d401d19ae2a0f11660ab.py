"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Use a known good sorting network for 18 wires if possible,
    # otherwise fall back to odd-even transposition.
    # For now, we'll implement a simple improvement: use the known 
    # optimal or near-optimal networks from literature if available.
    
    # Since we don't have a pre-computed optimal network for 18 wires,
    # we'll stick with odd-even transposition but verify it's correct.
    # The odd-even transposition network for N wires uses N*(N-1)/2 comparators.
    # For N=18, that's 18*17/2 = 153 comparators.
    
    # Let's try a slightly better approach: use the bitonic sort network
    # which has better depth but might use more comparators.
    # Actually, let's just return the odd-even transposition as it's known to work.
    
    # Generate odd-even transposition network
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    
    return comparators
# EVOLVE-BLOCK-END
