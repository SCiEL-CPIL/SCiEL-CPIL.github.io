"""
Size-efficient sorting network search for 17 wires.
Uses bitmask-based verification for speed, greedy removal, and local search.
"""
import time
import random
import numpy as np

N = 17
TOTAL_BOOL_INPUTS = 1 << N  # 131072
SEED_COMPARATORS = 136  # odd-even transposition for 17 wires
SEED_SCORE = SEED_COMPARATORS  # score = seed/value, but we minimize value

def _build_seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network_bitmask(comps):
    """
    Verify network sorts all 2^N Boolean inputs using bitmask simulation.
    Returns True if valid.
    """
    n = len(comps)
    if n == 0:
        # Empty network only valid for N=1
        return N <= 1
    
    # Pre-compute the effect of each comparator on all 2^N inputs
    # For each comparator (i, j), we need to know for each bitmask x
    # what the result is after comparing wires i and j.
    # 
    # Instead of simulating each input through all comparators (too slow),
    # we use the zero-one principle more efficiently:
    # A network is a sorting network iff for every k (0..N), it sorts
    # the input with k ones and N-k zeros (in any position).
    # But even that is C(N,k) cases per k, totaling 2^N.
    #
    # Better approach: simulate all 2^N inputs in parallel using numpy.
    # For each input bitmask (0 to 2^N-1), track the state of N wires.
    
    # We'll process inputs in batches to manage memory.
    # Each input is an integer 0..2^N-1.
    # State: for each input, we have N wire values (0 or 1).
    # We can represent this as a 2D array of shape (batch_size, N).
    
    batch_size = 1024
    total_inputs = 1 << N
    
    for start in range(0, total_inputs, batch_size):
        end = min(start + batch_size, total_inputs)
        # Create input vectors
        masks = np.arange(start, end, dtype=np.int32)
        # Extract wire values: wire i is bit i of the mask
        # state[input_idx, wire] = (masks[input_idx] >> wire) & 1
        wires = np.array([[(m >> w) & 1 for w in range(N)] for m in masks], dtype=np.int8)
        
        # Apply each comparator
        for i, j in comps:
            # For each input, compare wires i and j
            wi = wires[:, i]
            wj = wires[:, j]
            # min goes to i, max goes to j
            min_val = np.minimum(wi, wj)
            max_val = np.maximum(wi, wj)
            wires[:, i] = min_val
            wires[:, j] = max_val
        
        # Check if sorted: for each input, wires should be non-decreasing
        # i.e., wires[:, 0] <= wires[:, 1] <= ... <= wires[:, N-1]
        # This means all 0s come before all 1s
        # Equivalent: for each input, the sequence is sorted
        diffs = np.diff(wires, axis=1)
        if np.any(diffs < 0):
            return False
    
    return True

def _verify_network_fast(comps):
    """
    Faster verification using the fact that we only need to check
    that for every subset of positions of ones, the network sorts them.
    Uses numpy vectorization over all inputs at once if memory allows.
    """
    total_inputs = 1 << N
    if total_inputs > 200000:
        # Fall back to batched
        return _verify_network_bitmask(comps)
    
    masks = np.arange(total_inputs, dtype=np.int32)
    wires = np.array([[(m >> w) & 1 for w in range(N)] for m in masks], dtype=np.int8)
    
    for i, j in comps:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    diffs = np.diff(wires, axis=1)
    return not np.any(diffs < 0)

def _remove_redundant(comps, deadline):
    """Try to remove each comparator if the network remains valid."""
    n = len(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_fast(candidate):
                comps = candidate
                changed = True
    return comps

def _local_search(comps, deadline, max_iter=50):
    """
    Local search: try swapping adjacent comparators, or moving a comparator
    to a different position, if it maintains validity.
    """
    current = list(comps)
    best = list(current)
    best_len = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        improved = False
        n = len(current)
        # Try all possible single moves: swap adjacent, or move one comparator
        for i in range(n - 1):
            if time.time() > deadline - 10:
                break
            # Swap i and i+1
            candidate = current[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify_network_fast(candidate):
                current = candidate
                improved = True
        
        if not improved:
            # Try moving a comparator to a later position
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, min(i + 5, n)):
                    candidate = current[:]
                    elem = candidate.pop(i)
                    candidate.insert(j, elem)
                    if _verify_network_fast(candidate):
                        current = candidate
                        improved = True
                        break
        
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    # Start with seed network
    comps = _build_seed_network()
    best = list(comps)
    best_len = len(comps)
    
    # Phase 1: Greedy removal
    if time.time() < deadline - 20:
        removed = _remove_redundant(comps, deadline - 20)
        if len(removed) < best_len:
            best = removed
            best_len = len(removed)
        comps = removed
    
    # Phase 2: Local search
    if time.time() < deadline - 10:
        searched = _local_search(comps, deadline - 10, max_iter=30)
        if len(searched) < best_len:
            best = searched
            best_len = len(searched)
    
    return best

# Keep the original function name
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s kill
    
    # Start with seed network
    comps = _build_seed_network()
    best = list(comps)
    best_len = len(comps)
    
    # Phase 1: Greedy removal of redundant comparators
    if time.time() < deadline - 20:
        removed = _remove_redundant(comps, deadline - 20)
        if len(removed) < best_len:
            best = removed
            best_len = len(removed)
        comps = removed
    
    # Phase 2: Local search for further improvements
    if time.time() < deadline - 10:
        searched = _local_search(comps, deadline - 10, max_iter=30)
        if len(searched) < best_len:
            best = searched
            best_len = len(searched)
    
    return best
