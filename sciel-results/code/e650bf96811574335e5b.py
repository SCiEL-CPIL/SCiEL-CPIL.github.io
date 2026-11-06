"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            return False
    return True

def _remove_redundant(comparators, n=N, time_limit=55):
    """Try to remove redundant comparators using multiple strategies."""
    import random
    start_time = time.time()
    current = list(comparators)
    
    def try_remove(indices_to_try):
        nonlocal current
        # indices_to_try is a list of indices to try removing
        # We iterate through them, but if we remove one, the indices shift.
        # So we need to be careful. It's better to just try removing one at a time
        # and restart the scan if we succeed.
        for idx in indices_to_try:
            if time.time() - start_time > time_limit:
                return False
            if idx >= len(current):
                continue
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n):
                current = candidate
                return True
        return False

    while True:
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        L = len(current)
        if L == 0:
            break

        # Strategy 1: Forward pass
        if try_remove(range(L)):
            progress = True
        
        # Strategy 2: Backward pass
        if try_remove(range(L - 1, -1, -1)):
            progress = True
            
        # Strategy 3: Random pass
        indices = list(range(L))
        random.shuffle(indices)
        if try_remove(indices):
            progress = True

        if not progress:
            break
    
    return current

def construct():
    # Generate Batcher's network
    network = _batcher(N)
    
    # Try to remove redundant comparators
    network = _remove_redundant(network, N, time_limit=50)
    
    return network
# EVOLVE-BLOCK-END
