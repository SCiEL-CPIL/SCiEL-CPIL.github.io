"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _all_boolean_inputs(n):
    """Generate all 2^n Boolean inputs as a numpy array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.uint64)
    wires = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = (indices >> i) & 1
    return wires

def _verify_network(network, inputs):
    """Verify if the network sorts all given inputs."""
    if len(network) == 0:
        # Check if inputs are already sorted (they are not, generally)
        # But for efficiency, if network is empty, just check inputs
        return np.all(np.all(inputs[:, :-1] <= inputs[:, 1:], axis=1))
    
    # Copy inputs to avoid modifying the original
    state = inputs.copy()
    
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        # min to i, max to j
        # If a > b, swap
        mask = a > b
        if np.any(mask):
            state[:, i] = np.where(mask, b, a)
            state[:, j] = np.where(mask, a, b)
            
    # Check if sorted
    return np.all(np.all(state[:, :-1] <= state[:, 1:], axis=1))

def _batcher_odd_merge(n):
    """Generate Batcher's odd-merge sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    comparators = []
    n_odd = (n + 1) // 2
    n_even = n // 2
    
    # Sort odd and even positions
    sub_odd = _batcher_odd_merge(n_odd)
    for i, j in sub_odd:
        comparators.append((2*i, 2*j))
        
    sub_even = _batcher_odd_merge(n_even)
    for i, j in sub_even:
        comparators.append((2*i + 1, 2*j + 1))
        
    if n % 2 == 0:
        m = n // 2
        # Merge
        for i in range(0, m, 2):
            comparators.append((i, i + m))
            
        sub_merge_odd = _batcher_odd_merge(m)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
            
        sub_merge_even = _batcher_odd_merge(m)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
    else:
        m = n // 2
        # Merge
        for i in range(0, 2*m, 2):
            if i + m < n:
                comparators.append((i, i + m))
                
        sub_merge_odd = _batcher_odd_merge((n + 1) // 2)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
            
        sub_merge_even = _batcher_odd_merge(n // 2)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
            
    return comparators

def construct():
    start_time = time.time()
    deadline = 55.0 # 5s margin
    
    # Pre-generate all boolean inputs
    bool_inputs = _all_boolean_inputs(N)
    
    # Start with Batcher's Odd-Merge Sort
    network = _batcher_odd_merge(N)
    
    # Verify initial network (should be valid)
    if not _verify_network(network, bool_inputs):
        # Fallback to a known simple network if Batcher implementation is buggy
        # Odd-Even Transposition Sort
        network = []
        for phase in range(N):
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    
    # Greedy Deletion Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try to remove comparators from the end of the list to the beginning
        # This is often effective because later comparators are more redundant
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            
            # Candidate network without comparator at idx
            candidate = network[:idx] + network[idx+1:]
            
            # Verify
            if _verify_network(candidate, bool_inputs):
                network = candidate
                improved = True
                break # Restart the loop with the new network size
        
    return network
# EVOLVE-BLOCK-END
