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
    # Virtual indices 0..k_half-1 map to 0, 2, ..., k-2
    # Virtual indices k_half..k-1 map to k, k+2, ..., 2k-2
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    # Virtual indices 0..k_half-1 map to 1, 3, ..., k-1
    # Virtual indices k_half..k-1 map to k+1, k+3, ..., 2k-1
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
    # Sort the two halves
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    
    # Merge the two sorted halves
    merge = _merge(k)
    
    return left + right + merge

def _verify(comps):
    """
    Verify the network using the zero-one principle.
    Returns True if the network sorts all 2^N binary inputs.
    """
    n = len(comps)
    if n == 0:
        return True
    
    # Use numpy for vectorized verification
    # Create all 2^N masks
    masks = np.arange(1 << N, dtype=np.uint8)
    
    # Reshape to (65536, N) where each row is a binary input
    # wire i gets bit i of the mask
    wires = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        wires[:, i] = (masks >> i) & 1
    
    # Apply comparators
    for i, j in comps:
        # For each wire pair, swap if wires[i] > wires[j]
        # Since values are 0/1, swap means:
        # new_wires[i] = min(wires[i], wires[j]) = wires[i] & wires[j]
        # new_wires[j] = max(wires[i], wires[j]) = wires[i] | wires[j]
        wi = wires[:, i].copy()
        wj = wires[:, j].copy()
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if for all i, wires[i] <= wires[i+1]
    # i.e., no adjacent pair has wires[i] > wires[i+1]
    for i in range(N - 1):
        if np.any(wires[:, i] > wires[:, i + 1]):
            return False
    
    return True

def construct():
    """
    Construct a sorting network for N=16 wires.
    Returns a list of (i, j) comparator pairs.
    """
    # Start with Batcher's odd-even merge sort (63 comparators)
    comps = _batcher(N)
    
    # Verify correctness
    if _verify(comps):
        # Try to optimize by removing redundant comparators
        # We'll use a greedy approach: try removing each comparator and check
        # if the network is still correct.
        
        # Time limit guard
        start_time = time.time()
        deadline = start_time + 85.0  # 85 seconds to leave 5s margin
        
        # Greedy removal: iterate through comparators and try to remove each
        # We process in reverse order (later comparators are more likely redundant)
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            trial = comps[:idx] + comps[idx+1:]
            
            if _verify(trial):
                comps = trial
        
        return comps
    
    # Fallback: odd-even transposition
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
