"""Size-efficient sorting networks on 16 wires.

Strategy:
1. Generate a correct Batcher's Odd-Even Mergesort network for N=16 (60 comparators).
2. Use a highly optimized NumPy-based verifier for the Zero-One Principle.
3. Perform local search (greedy removal + random restarts/swaps) to minimize comparator count.
4. Return the best valid network found within the time budget.
"""

import time
import numpy as np
import random

N = 16
DEADLINE = 90  # seconds
SAFETY_MARGIN = 5  # seconds

# Precompute all 2^16 binary inputs for fast verification
# Shape: (65536, 16), dtype: uint8
# inputs[i, j] is bit j of input i
_ALL_INPUTS = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    _ALL_INPUTS[:, i] = (np.arange(1 << N) >> i) & 1

def _verify_network(net):
    """
    Verify that the network sorts all 16-input sequences using the zero-one principle.
    Uses vectorized NumPy operations for speed.
    Returns True if the network is a valid sorting network.
    """
    if not net:
        return N <= 1
    
    # Copy inputs to avoid modifying the global array
    bits = _ALL_INPUTS.copy()
    
    # Apply the network
    for (a, b) in net:
        # min goes to a, max goes to b
        # We need to update bits in place
        min_val = np.minimum(bits[:, a], bits[:, b])
        max_val = np.maximum(bits[:, a], bits[:, b])
        bits[:, a] = min_val
        bits[:, b] = max_val
    
    # Check if sorted: bits should be in non-decreasing order along wires
    # i.e., for each row, bits[0] <= bits[1] <= ... <= bits[N-1]
    # Since bits are 0/1, sorted means all 0s come before all 1s.
    # Check adjacent differences: no -1 allowed.
    diffs = np.diff(bits, axis=1)
    return not np.any(diffs < 0)

def _batcher_oes(n):
    """
    Generate Batcher's Odd-Even Mergesort network for n inputs.
    Returns a list of comparators (i, j) with 0 <= i < j < n.
    """
    if n == 1:
        return []
    
    half = n // 2
    # Recursively sort the two halves
    left = _batcher_oes(half)
    right = _batcher_oes(half)
    
    # Shift right comparators by half
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two sorted halves
    merge = _batcher_merge(half)
    
    return left + right + merge

def _batcher_merge(k):
    """
    Generate the merge phase for two sorted lists of size k.
    Total wires: 2k.
    The inputs are assumed to be:
    - First half (wires 0..k-1) sorted ascending
    - Second half (wires k..2k-1) sorted ascending
    
    Returns comparators to merge these into a single sorted sequence of size 2k.
    """
    if k == 1:
        return [(0, 1)]
    
    comparators = []
    
    # Recursive merge structure:
    # 1. Recursively merge the even-indexed elements of the two halves
    # 2. Recursively merge the odd-indexed elements of the two halves
    # 3. Compare and swap adjacent pairs (1,2), (3,4), ..., (2k-1, 2k)
    
    # Even indices in first half: 0, 2, ..., k-2 (if k even)
    # Even indices in second half: k, k+2, ..., 2k-2
    # These form two sorted sub-sequences of size k//2 (if k is even)
    # However, Batcher's merge is defined for any k, but typically k is a power of 2.
    # For k=2:
    #   Evens: 0, 2 -> merge(1) -> (0, 2)
    #   Odds: 1, 3 -> merge(1) -> (1, 3)
    #   Cross: (1, 2)
    #   Total: (0,2), (1,3), (1,2)
    
    # Let's implement the standard recursive definition carefully.
    # The merge of two sorted arrays of size k:
    # M(k) = M(k/2) on evens + M(k/2) on odds + Cross comparisons
    
    # But the indices are interleaved.
    # Let's use a helper that generates comparators for a specific "stride" or "level".
    
    # Alternative robust implementation:
    # Batcher's merge of two sorted lists of size n:
    # 1. Recursively merge the elements at even positions: A[0], A[2]... and B[0], B[2]...
    # 2. Recursively merge the elements at odd positions: A[1], A[3]... and B[1], B[3]...
    # 3. Compare (1,2), (3,4), ..., (2n-1, 2n)
    
    # Since our wires are contiguous [0..k-1] and [k..2k-1], 
    # the "even positions" of the combined logical array aren't simply contiguous wires.
    # We need to map logical positions to physical wires.
    
    # Let's trace for k=2 (total 4 wires: 0,1,2,3)
    # Left: 0,1 (sorted). Right: 2,3 (sorted).
    # Evens: 0, 2. Merge(1) on {0,2} -> (0,2).
    # Odds: 1, 3. Merge(1) on {1,3} -> (1,3).
    # Cross: (1,2).
    # Result: (0,2), (1,3), (1,2).
    
    # Let's trace for k=4 (total 8 wires: 0..7)
    # Left: 0..3. Right: 4..7.
    # Evens: 0, 2, 4, 6. 
    #   These are not contiguous. We need to merge {0,2} with {4,6}?
    #   No, the recursive merge operates on the *values* at these positions.
    #   To use the same function, we need to treat the wires {0,2,4,6} as a sub-network.
    #   But our function assumes contiguous wires.
    
    # Therefore, we need a more general merge function that takes a list of wire indices.
    
    def merge_on_indices(indices):
        """
        Merge two sorted sequences located at indices[0:k] and indices[k:2k].
        indices must be a list of 2k wire numbers.
        """
        k_local = len(indices) // 2
        if k_local == 1:
            return [(indices[0], indices[1])]
        
        # Split indices into even and odd positions in the logical sequence
        # Logical even positions: 0, 2, ...
        # Logical odd positions: 1, 3, ...
        
        even_indices = []
        odd_indices = []
        for i in range(len(indices)):
            if i % 2 == 0:
                even_indices.append(indices[i])
            else:
                odd_indices.append(indices[i])
        
        # Recursively merge even and odd parts
        comps_even = merge_on_indices(even_indices)
        comps_odd = merge_on_indices(odd_indices)
        
        # Cross comparisons: (1,2), (3,4), ... in logical indices
        # Logical index 1 is indices[1], Logical index 2 is indices[2]
        cross_comps = []
        for i in range(1, len(indices), 2):
            # Compare logical i and i+1
            # i is odd, i+1 is even
            a = indices[i]
            b = indices[i+1]
            if a < b:
                cross_comps.append((a, b))
            else:
                cross_comps.append((b, a))
        
        return comps_even + comps_odd + cross_comps
    
    # Initial indices for the top-level merge of two halves of size k
    # First half wires: 0, 1, ..., k-1
    # Second half wires: k, k+1, ..., 2k-1
    all_indices = list(range(2 * k))
    
    return merge_on_indices(all_indices)

def _generate_batcher_network(n):
    """Generate the full Batcher network for n wires."""
    return _batcher_oes(n)

def _local_search(net, time_limit):
    """
    Local search to minimize comparator count.
    1. Greedy removal.
    2. Random swaps and restarts if stuck.
    """
    start_time = time.time()
    current_net = list(net)
    best_net = list(current_net)
    best_count = len(current_net)
    
    # Verify initial net
    if not _verify_network(current_net):
        return best_net # Should not happen if generated correctly
    
    # 1. Greedy Removal Pass
    changed = True
    while changed:
        if time.time() - start_time > time_limit - SAFETY_MARGIN:
            break
        changed = False
        for idx in range(len(current_net)):
            if time.time() - start_time > time_limit - SAFETY_MARGIN:
                break
            test_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(test_net):
                current_net = test_net
                best_net = list(current_net)
                best_count = len(current_net)
                changed = True
                break # Restart loop as indices shifted
    
    # 2. Random Restart / Swap Search
    # Try to find better configurations by swapping comparators or removing/adding
    # This is more expensive, so we limit iterations based on time.
    
    # We can also try "reordering" comparators that are independent to enable removal.
    # However, with the fast verifier, we can just try random perturbations.
    
    while time.time() - start_time < time_limit - SAFETY_MARGIN:
        # Randomly swap two comparators
        if len(current_net) < 2:
            break
        
        i = random.randint(0, len(current_net) - 1)
        j = random.randint(0, len(current_net) - 1)
        if i == j:
            continue
            
        test_net = list(current_net)
        test_net[i], test_net[j] = test_net[j], test_net[i]
        
        # Check if valid
        if _verify_network(test_net):
            # Try to remove a random comparator from the swapped network
            if len(test_net) > 0:
                rem_idx = random.randint(0, len(test_net) - 1)
                test_net2 = test_net[:rem_idx] + test_net[rem_idx+1:]
                if _verify_network(test_net2):
                    current_net = test_net2
                    if len(current_net) < best_count:
                        best_net = list(current_net)
                        best_count = len(current_net)
            
            # Also just accept the swap if it doesn't increase count (it doesn't)
            # But we only care about count reduction.
            # So we only update best if we found a removal.
            
            # Occasionally, try to remove a comparator from the current net without swap
            # (already done in greedy, but maybe random order helps?)
            # Let's stick to swap-then-remove.
            
        # Additional move: Try removing a random comparator from current_net
        # (This is similar to greedy but random order)
        if len(current_net) > 0 and random.random() < 0.5:
            rem_idx = random.randint(0, len(current_net) - 1)
            test_net = current_net[:rem_idx] + current_net[rem_idx+1:]
            if _verify_network(test_net):
                current_net = test_net
                if len(current_net) < best_count:
                    best_net = list(current_net)
                    best_count = len(current_net)
                    
    return best_net

def construct():
    """
    Construct a sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j), 0 <= i < j < 16.
    """
    start_time = time.time()
    time_budget = 90 - SAFETY_MARGIN
    
    # 1. Generate Batcher's Odd-Even Mergesort Network
    # This should yield 60 comparators for N=16
    net = _generate_batcher_network(N)
    
    # Verify it
    if not _verify_network(net):
        # Fallback to Odd-Even Transposition if Batcher generation failed
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # 2. Local Search
    if time.time() - start_time < time_budget:
        net = _local_search(net, time_budget)
    
    # 3. Final Safety Check
    if time.time() - start_time < 90 - 2:
        if not _verify_network(net):
            # Fall back to known correct network
            net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    return net

