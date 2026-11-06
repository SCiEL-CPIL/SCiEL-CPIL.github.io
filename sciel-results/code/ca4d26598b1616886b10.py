"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _apply_network(wires, comparators):
    """Apply a sorting network to an array of wires."""
    for i, j in comparators:
        a, b = wires[i], wires[j]
        if a > b:
            wires[i], wires[j] = b, a
    return wires

def _is_sorting_network(comparators):
    """Check if network sorts all 2^18 Boolean inputs using 0-1 principle."""
    n = len(comparators)
    if n > 4096:
        return False
    # Use 0-1 principle: only need to check Boolean inputs
    # For efficiency, use numpy vectorized operations
    for mask in range(1, 1 << N):
        wires = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in comparators:
            a, b = wires[i], wires[j]
            if a > b:
                wires[i], wires[j] = b, a
        # Check if sorted (non-decreasing)
        if np.any(np.diff(wires) < 0):
            return False
    return True

def _try_remove(comparators, index):
    """Try removing comparator at index; return True if still sorting."""
    reduced = comparators[:index] + comparators[index+1:]
    return _is_sorting_network(reduced)

def _greedy_reduce(comparators, deadline):
    """Greedy reduction: try removing each comparator, keep if valid."""
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        if _try_remove(current, i):
            current = current[:i] + current[i+1:]
            # Don't increment i, check same position again
        else:
            i += 1
    return current

def _randomize_order(comparators):
    """Shuffle comparator order to escape local minima."""
    shuffled = list(comparators)
    random.shuffle(shuffled)
    return shuffled

def construct():
    # Start with odd-even transposition network
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    best = list(seed_network)
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin for 90s limit
    
    # Multiple random restarts with greedy reduction
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the order to explore different reduction paths
        randomized = _randomize_order(seed_network)
        
        # Apply greedy reduction
        reduced = _greedy_reduce(randomized, deadline)
        
        if len(reduced) < len(best):
            best = reduced
    
    return best
# EVOLVE-BLOCK-END
