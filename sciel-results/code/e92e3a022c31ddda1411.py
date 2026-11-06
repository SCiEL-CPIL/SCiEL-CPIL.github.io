"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a size-efficient sorting network for 16 wires.
    Uses a heuristic search to minimize comparator count
    while verifying correctness via the zero-one principle.
    
    Optimized with NumPy for fast verification of all 2^16 inputs.
    """
    
    def verify(comparators):
        # Check all 2^16 binary inputs using NumPy vectorization
        # inputs is an array of all 16-bit integers
        inputs = np.arange(1 << N, dtype=np.uint16)
        
        for i, j in comparators:
            mask_i = 1 << i
            mask_j = 1 << j
            # Condition: bit i is 1 AND bit j is 0
            # We swap if this condition is true
            cond = ((inputs & mask_i) != 0) & ((inputs & mask_j) == 0)
            if np.any(cond):
                inputs[cond] ^= (mask_i | mask_j)
        
        # Check if output is sorted
        # A sorted binary vector 0...01...1 corresponds to an integer x where x+1 is a power of 2.
        # i.e., (x + 1) & x == 0
        # Note: np.uint16 arithmetic wraps around, but for x=65535 (all 1s), x+1=0 (wrap), 0 & 65535 = 0. Correct.
        # For x=0, x+1=1, 1 & 0 = 0. Correct.
        valid = ((inputs + 1) & inputs) == 0
        return np.all(valid)

    # Start with the odd-even transposition network (120 comparators)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy reduction with multiple passes
    # We use a time-based loop or a fixed number of passes to ensure we finish within 60s
    # Since verify is fast, we can do many passes.
    max_passes = 50
    for _ in range(max_passes):
        removed_any = False
        # Shuffle the indices to avoid bias
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if idx >= len(network):
                continue
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if verify(candidate):
                network = candidate
                removed_any = True
        if not removed_any:
            break
            
    # If we still have a large network, try some local search moves (add/remove) to escape local minima
    # This is optional but can help find smaller networks.
    # Given time constraints, we stick to the robust greedy removal which is guaranteed to improve or stay same.
    
    return network
# EVOLVE-BLOCK-END
