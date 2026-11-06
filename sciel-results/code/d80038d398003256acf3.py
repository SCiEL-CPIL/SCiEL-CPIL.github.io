import numpy as np
import time

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires.
    1. Generates an initial valid network using Batcher's Merge Sort.
    2. Optimizes the network by greedily removing redundant comparators
       using vectorized simulation over all 2^N boolean inputs.
    """
    
    def batcher_merge_sort(n):
        if n <= 1:
            return []
        
        k = n // 2
        left = batcher_merge_sort(k)
        right = [(i + k, j + k) for (i, j) in batcher_merge_sort(n - k)]
        merge = batcher_odd_merge(k, n - k)
        
        return left + right + merge
    
    def batcher_odd_merge(n, m):
        if n == 0 or m == 0:
            return []
        if n == 1 and m == 1:
            return [(0, 1)]
        
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        merge_odd = batcher_odd_merge(n_odd, m_even)
        def map_odd(pos):
            if pos < n_odd:
                return 2 * pos + 1
            else:
                return n + 2 * (pos - n_odd)
        mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
        
        merge_even = batcher_odd_merge(n_even, m_odd)
        def map_even(pos):
            if pos < n_even:
                return 2 * pos
            else:
                return n + 2 * (pos - n_even) + 1
        mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
        
        total = n + m
        final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
        
        return mapped_odd + mapped_even + final_comps

    # Generate initial network
    network = batcher_merge_sort(N)
    
    # Optimization Phase
    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer for safety
    
    # Pre-compute the truth table for all 2^N boolean inputs
    # Shape: (2^N, N)
    # dtype: bool
    truth_table = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)], dtype=np.bool_)
    
    def is_sorting(comparators):
        """
        Checks if the given comparators sort all boolean inputs.
        Uses vectorized operations.
        """
        # Copy the truth table to avoid modifying the original
        states = truth_table.copy()
        
        for i, j in comparators:
            # Apply compare-exchange
            min_val = np.minimum(states[:, i], states[:, j])
            max_val = np.maximum(states[:, i], states[:, j])
            states[:, i] = min_val
            states[:, j] = max_val
        
        # Check if all rows are sorted (non-decreasing)
        # Compare adjacent columns
        # states[:, :-1] <= states[:, 1:]
        sorted_flags = np.all(states[:, :-1] <= states[:, 1:], axis=1)
        return np.all(sorted_flags)

    # Greedy Reduction
    # We iterate multiple passes until no more comparators can be removed
    # or time limit is reached.
    
    # To speed up, we can try removing from the end first, as later comparators
    # are often more critical for final sorting, but earlier ones might be redundant
    # due to previous sorting. Actually, redundancy is often found in the middle.
    # We'll just iterate through all.
    
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        # Iterate backwards to handle index removal safely
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Create candidate network without network[idx]
            # This is O(M) copy. M is small (~80).
            candidate = network[:idx] + network[idx+1:]
            
            # Check validity
            if is_sorting(candidate):
                network = candidate
                changed = True
    
    return network

# EVOLVE-BLOCK-END
