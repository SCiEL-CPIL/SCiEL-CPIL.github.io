import numpy as np
import time
import math
import random

N = 18

def _batcher_merge(N1, N2):
    if N1 == 0 or N2 == 0:
        return []
    depth = math.ceil(math.log2(N1 + N2))
    comps = []
    for j in range(depth):
        step = 1 << j
        for i in range(N1 + N2 - step):
            if j == 0:
                comps.append((i, i + step))
            else:
                if (i % step) < (step // 2):
                    comps.append((i, i + step))
    return comps

def _batcher_sort(N):
    if N == 1:
        return []
    N1 = N // 2
    N2 = N - N1
    comps = []
    comps.extend(_batcher_sort(N1))
    half2_comps = _batcher_sort(N2)
    comps.extend([(i + N1, j + N1) for i, j in half2_comps])
    comps.extend(_batcher_merge(N1, N2))
    return comps

def _verify_batch(comps_list, N):
    """
    Verify multiple networks in parallel using bit-parallel NumPy.
    comps_list: list of lists of (i,j) pairs
    Returns: list of booleans, True if network sorts all inputs
    """
    if not comps_list:
        return []
    
    n_networks = len(comps_list)
    max_comps = max(len(c) for c in comps_list)
    
    # Precompute all inputs
    inputs = np.arange(2**N, dtype=np.uint32)
    bits = 1 << np.arange(N)
    base_matrix = (inputs[:, None] & bits[None, :]) > 0
    
    results = []
    
    for comps in comps_list:
        # Copy the base matrix
        matrix = base_matrix.copy()
        
        # Apply network
        for i, j in comps:
            min_val = np.minimum(matrix[:, i], matrix[:, j])
            max_val = np.maximum(matrix[:, i], matrix[:, j])
            matrix[:, i] = min_val
            matrix[:, j] = max_val
            
        # Check if sorted
        diff = np.diff(matrix, axis=1)
        results.append(bool(np.all(diff >= 0)))
    
    return results

def _local_search(comps, N, deadline):
    """
    Local search: try removing or swapping comparators.
    Uses batched verification for efficiency.
    """
    best_comps = list(comps)
    best_count = len(comps)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Try random removals in batches
        batch_size = min(20, len(best_comps))
        if batch_size == 0:
            break
        
        candidates = []
        candidate_indices = []
        
        # Try removing random comparators
        for _ in range(batch_size):
            idx = random.randint(0, len(best_comps) - 1)
            if idx not in candidate_indices:
                candidate_indices.append(idx)
                test = best_comps[:idx] + best_comps[idx+1:]
                candidates.append(test)
        
        # Try swapping with random comparator
        if len(best_comps) > 1:
            for _ in range(batch_size // 2):
                idx = random.randint(0, len(best_comps) - 1)
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                new_comp = (new_i, new_j)
                
                # Avoid duplicates
                if new_comp in best_comps:
                    continue
                    
                test = list(best_comps)
                test[idx] = new_comp
                # Sort by depth for consistency (optional)
                candidates.append(test)
        
        if not candidates:
            break
        
        # Verify batch
        results = _verify_batch(candidates, N)
        
        improved = False
        for i, (cand, res) in enumerate(zip(candidates, results)):
            if res:
                if len(cand) < best_count:
                    best_comps = cand
                    best_count = len(cand)
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_comps

def _greedy_prune(comps, N, deadline):
    """
    Greedy pruning: try to remove each comparator if possible.
    """
    best_comps = list(comps)
    i = 0
    while i < len(best_comps) and time.time() < deadline:
        test = best_comps[:i] + best_comps[i+1:]
        if _verify_batch([test], N)[0]:
            best_comps = test
            # Don't increment i
        else:
            i += 1
    return best_comps

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    # 1. Generate Batcher network as starting point
    try:
        comps = _batcher_sort(N)
        comps = list(set(comps))
    except Exception:
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
    
    best_comps = list(comps)
    best_count = len(comps)
    
    # 2. Greedy pruning phase
    if time.time() < deadline - 30:
        pruned = _greedy_prune(comps, N, deadline - 30)
        if len(pruned) < best_count:
            best_comps = pruned
            best_count = len(pruned)
    
    # 3. Local search phase
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        searched = _local_search(best_comps, N, deadline)
        if len(searched) < best_count:
            best_comps = searched
            best_count = len(searched)
    
    # Final safety check
    if not best_comps:
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
        best_comps = comps
    
    return best_comps
