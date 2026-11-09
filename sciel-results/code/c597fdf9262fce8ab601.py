"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for 17 wires.
    Uses a known small network if available, otherwise falls back to a
    reasonably efficient construction.
    
    The odd-even transposition network for 17 wires uses 17*16/2 = 136 comparators.
    We can try to do better with a simple heuristic or known construction.
    
    For now, let's use a slightly optimized approach:
    - Use bitonic sort which for 16 wires is 60 comparators, but for 17 we need to handle the extra wire.
    - Alternatively, we can use a known smaller network.
    
    Actually, let's just use the odd-even transposition but verify it's correct.
    For N=17, odd-even transposition takes N phases, each with ~N/2 comparators.
    Total: 17 * 8 = 136 (approximately, since some phases have 8 and some have 9).
    
    Let me count exactly:
    Phase 0: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15), (16,17) -> wait, 17 wires means indices 0-16.
    So phase 0: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15) -> 8 comparators
    Phase 1: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16) -> 8 comparators
    Phase 2: same as phase 0 -> 8
    ...
    So 17 phases * 8 = 136 comparators.
    
    Let's try to use a better network. A simple approach: use a known 16-wire network plus handling for the 17th wire.
    
    For 16 wires, bitonic sort uses 60 comparators. We can merge the 17th wire.
    
    Actually, let's just return the odd-even transposition for now as a baseline, and the evolution can improve it.
    """
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
