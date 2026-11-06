"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17

# EVOLVE-BLOCK-START
import numpy as np
import time

def _all_bool_inputs(n):
    """Generate all 2^n boolean inputs as a (2^n, n) array of 0s and 1s."""
    return np.array([[int((mask >> i) & 1) for i in range(n)] for mask in range(1 << n)], dtype=np.int8)

def _is_sorted_network(comparators, inputs):
    """Check if the network sorts all given inputs using zero-one principle."""
    n = inputs.shape[1]
    # Process all inputs in parallel
    state = inputs.copy()
    for (i, j) in comparators:
        # compare-exchange: min to i, max to j
        col_i = state[:, i]
        col_j = state[:, j]
        new_i = np.minimum(col_i, col_j)
        new_j = np.maximum(col_i, col_j)
        state[:, i] = new_i
        state[:, j] = new_j
    # Check if all outputs are sorted (non-decreasing)
    # For a sorted array, each element should be >= previous
    for k in range(1, n):
        if np.any(state[:, k] < state[:, k - 1]):
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Pre-compute all boolean inputs
    all_inputs = _all_bool_inputs(N)
    
    # Start with seed network
    network = _seed_network()
    
    # Local search: try to remove each comparator
    # Iterate until no more removals possible or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
                improved = True
                break  # Restart from beginning since indices shifted
    
    return network
# EVOLVE-BLOCK-END
