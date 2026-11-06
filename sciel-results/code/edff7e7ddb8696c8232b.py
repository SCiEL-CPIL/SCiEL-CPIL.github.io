import sys
import numpy as np
import itertools

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires using Batcher's Merge Sort,
    then optimizes it by greedily removing redundant comparators.
    """
    
    def batcher_merge_sort(n):
        """Returns a list of comparators for sorting n wires using Batcher's Merge Sort."""
        if n <= 1:
            return []
        
        k = n // 2
        left = batcher_merge_sort(k)
        right = [(i + k, j + k) for (i, j) in batcher_merge_sort(n - k)]
        merge = batcher_odd_merge(k, n - k)
        
        return left + right + merge
    
    def batcher_odd_merge(n, m):
        """
        Returns comparators to merge two sorted lists of sizes n and m.
        The first list occupies indices 0..n-1, the second occupies indices n..n+m-1.
        Uses Batcher's Odd-Merge algorithm.
        """
        if n == 0 or m == 0:
            return []
        
        if n == 1 and m == 1:
            return [(0, 1)]
        
        # Use the standard Batcher Odd-Merge algorithm
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        # Merge A_odd (size n_odd) with B_even (size m_even)
        merge_odd = batcher_odd_merge(n_odd, m_even)
        
        def map_odd(pos):
            if pos < n_odd:
                return 2 * pos + 1
            else:
                return n + 2 * (pos - n_odd)
        
        mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
        
        # Merge A_even (size n_even) with B_odd (size m_odd)
        merge_even = batcher_odd_merge(n_even, m_odd)
        
        def map_even(pos):
            if pos < n_even:
                return 2 * pos
            else:
                return n + 2 * (pos - n_even) + 1
        
        mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
        
        # Final comparators
        total = n + m
        final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
        
        return mapped_odd + mapped_even + final_comps

    # Generate all 2^N binary inputs
    # Shape: (2^N, N)
    # Using numpy for vectorized simulation
    
    # Create all combinations of 0s and 1s
    # It is more efficient to use bit manipulation or product
    # 2^18 is 262144. 
    # We can create the array using np.arange and bit extraction?
    # Or just use itertools.product and convert to numpy array.
    
    # For N=18, itertools.product is fine.
    # But we can generate it faster with numpy.
    # Let's use the standard method for 0-1 principle verification.
    
    # Initialize the input matrix
    # We want all binary vectors of length N.
    # rows = 2^N
    rows = 1 << N
    # Create array of indices 0 to 2^N - 1
    indices = np.arange(rows, dtype=np.uint32)
    
    # Expand to bits
    # For each bit position b (0 to N-1), the bit is (indices >> b) & 1
    # We want shape (rows, N)
    # bits = (indices[:, None] >> np.arange(N)) & 1
    M = (indices[:, None] >> np.arange(N)) & 1
    M = M.astype(np.uint8)
    
    # Initial network
    network = batcher_merge_sort(N)
    
    # Pre-compute states after each prefix of the network
    # states[k] is the matrix after applying network[0:k]
    # states[0] is M
    states = [M]
    curr_state = M.copy()
    
    for comp in network:
        i, j = comp
        mask = curr_state[:, i] > curr_state[:, j]
        # Swap
        curr_state[:, i][mask], curr_state[:, j][mask] = curr_state[:, j][mask], curr_state[:, i][mask]
        states.append(curr_state.copy())
        
    # Check if final state is sorted
    def is_sorted(state):
        return np.all(state[:, :-1] <= state[:, 1:])
    
    # Greedy removal
    # We iterate through the network and try to remove each comparator
    # If removing it keeps the network valid, we remove it and restart the scan
    
    # To speed up, we check validity by simulating from the state before the comparator
    # to the end, using the remaining comparators.
    
    L = len(network)
    changed = True
    while changed:
        changed = False
        for idx in range(L):
            # Try to remove network[idx]
            # We need to check if network without idx is valid
            # State before idx is states[idx]
            # We apply network[idx+1:] to a copy of states[idx]
            
            if idx >= L:
                break
                
            test_state = states[idx].copy()
            
            # Apply remaining comparators
            valid = True
            for k in range(idx + 1, L):
                i, j = network[k]
                mask = test_state[:, i] > test_state[:, j]
                test_state[:, i][mask], test_state[:, j][mask] = test_state[:, j][mask], test_state[:, i][mask]
            
            if is_sorted(test_state):
                # Removal is valid
                # Update network
                network.pop(idx)
                L -= 1
                changed = True
                # Re-compute states? 
                # Yes, states are invalid now.
                # We can re-compute states for the new network.
                # Or just break and let the outer loop re-compute?
                # Re-computing states is O(L * Cost).
                # It's better to re-compute states after a change.
                
                # Re-compute states
                states = [M]
                curr_state = M.copy()
                for comp in network:
                    i, j = comp
                    mask = curr_state[:, i] > curr_state[:, j]
                    curr_state[:, i][mask], curr_state[:, j][mask] = curr_state[:, j][mask], curr_state[:, i][mask]
                    states.append(curr_state.copy())
                
                break # Restart the for loop
        
    return network
