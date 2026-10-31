"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Multi-start greedy with randomized tie-breaking + aggressive local search
with removal, swap, and 2-swap moves using NumPy-vectorized coverage checks.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 118.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets, k-subsets, and coverage structures."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build coverage matrix (N_k x N_t) as boolean numpy array
    cov_matrix = np.zeros((n_k, n_t), dtype=np.uint8)
    
    # For CSR: indptr and indices
    sizes = np.zeros(n_k, dtype=np.int32)
    all_indices = []
    
    for i, ks in enumerate(k_subsets):
        ts_list = []
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            cov_matrix[i, idx] = 1
            ts_list.append(idx)
        sizes[i] = len(ts_list)
        all_indices.append(ts_list)
        
    indptr = np.zeros(n_k + 1, dtype=np.int32)
    np.cumsum(sizes, out=indptr[1:])
    indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    
    # Build inverted index: for each t-subset, list of k-blocks containing it
    # This is useful for local search
    inv_indptr = np.zeros(n_t + 1, dtype=np.int32)
    for i in range(n_k):
        start = indptr[i]
        end = indptr[i+1]
        for idx in indices[start:end]:
            inv_indptr[idx+1] += 1
    np.cumsum(inv_indptr, out=inv_indptr[1:])
    inv_indices = np.zeros(inv_indptr[-1], dtype=np.int32)
    temp = inv_indptr[:-1].copy()
    for i in range(n_k):
        start = indptr[i]
        end = indptr[i+1]
        for pos in range(start, end):
            idx = indices[pos]
            inv_indices[temp[idx]] = i
            temp[idx] += 1
    
    return all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k, inv_indptr, inv_indices


def _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k, seed_offset=0):
    """Greedy set cover using NumPy vectorization with randomized tie-breaking."""
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    rng_local = random.Random(12345 + seed_offset)
    
    while True:
        if _elapsed() > _DEADLINE - 30:
            break
        
        # Calculate gain for each block
        gains = np.sum(cov_matrix * uncovered, axis=1)
        
        best_gain = gains.max()
        if best_gain == 0:
            break
            
        # Randomized tie-breaking among blocks with max gain
        best_indices = np.where(gains == best_gain)[0]
        best_i = best_indices[rng_local.randrange(len(best_indices))]
        
        blocks.append(best_i)
        uncovered[cov_matrix[best_i] == 1] = 0
        
        if np.sum(uncovered) == 0:
            break
            
    return blocks


def _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, inv_indptr, inv_indices, rng, time_budget):
    """Local search using CSR for fast index retrieval and NumPy for counts."""
    blocks = list(blocks)
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    def is_valid(counts):
        return np.all(counts > 0)
        
    if not is_valid(counts):
        # Repair
        while not is_valid(counts) and _elapsed() < time_budget:
            missing = np.where(counts == 0)[0]
            if len(missing) == 0: break
            gains = np.sum(cov_matrix[:, missing], axis=1)
            best_i = np.argmax(gains)
            if gains[best_i] == 0: break
            blocks.append(best_i)
            start = indptr[best_i]
            end = indptr[best_i+1]
            if end > start:
                counts[indices[start:end]] += 1
                
    improved = True
    stall_count = 0
    n_blocks = len(blocks)
    best_n_blocks = n_blocks
    best_blocks = list(blocks)
    
    while improved and _elapsed() < time_budget:
        improved = False
        stall_count += 1
        
        # --- Removal pass ---
        removable_found = False
        for i in range(n_blocks):
            b = blocks[i]
            start = indptr[b]
            end = indptr[b+1]
            if end == start: continue
            if counts[indices[start:end]].min() > 1:
                counts[indices[start:end]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                removable_found = True
                if n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = list(blocks)
                break
        
        if removable_found:
            continue
            
        # --- Single swap pass ---
        for _ in range(100):
            if _elapsed() >= time_budget:
                break
            if n_blocks == 0:
                break
                
            i = rng.randrange(n_blocks)
            j = rng.randrange(n_k)
            if j in blocks:
                continue
                
            old = blocks[i]
            
            # Apply swap
            blocks[i] = j
            
            start_old = indptr[old]
            end_old = indptr[old+1]
            if end_old > start_old:
                counts[indices[start_old:end_old]] -= 1
            start_new = indptr[j]
            end_new = indptr[j+1]
            if end_new > start_new:
                counts[indices[start_new:end_new]] += 1
                
            if not is_valid(counts):
                counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old
                continue
                
            # Try removal after swap
            start_j = indptr[j]
            end_j = indptr[j+1]
            if end_j > start_j and counts[indices[start_j:end_j]].min() > 1:
                counts[indices[start_j:end_j]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                if n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = list(blocks)
                continue
                
            # Also check if removing old block's coverage makes something else removable
            # Check a few random blocks
            for _check in range(5):
                ci = rng.randrange(n_blocks)
                cb = blocks[ci]
                cs = indptr[cb]
                ce = indptr[cb+1]
                if ce > cs and counts[indices[cs:ce]].min() > 1:
                    counts[indices[cs:ce]] -= 1
                    blocks.pop(ci)
                    n_blocks -= 1
                    improved = True
                    stall_count = 0
                    if n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_blocks = list(blocks)
                    break
                    
            if not improved:
                # Revert swap
                counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old

        # --- Perturbation if stalled ---
        if not improved and stall_count > 3 and n_blocks > 10:
            if _elapsed() < time_budget - 10:
                i = rng.randrange(n_blocks)
                removed_block = blocks.pop(i)
                start = indptr[removed_block]
                end = indptr[removed_block+1]
                if end > start:
                    counts[indices[start:end]] -= 1
                n_blocks -= 1
                while not is_valid(counts) and _elapsed() < time_budget - 5:
                    missing = np.where(counts == 0)[0]
                    if len(missing) == 0: break
                    gains = np.sum(cov_matrix[:, missing], axis=1)
                    best_i = np.argmax(gains)
                    if gains[best_i] == 0: break
                    blocks.append(best_i)
                    start = indptr[best_i]
                    end = indptr[best_i+1]
                    if end > start:
                        counts[indices[start:end]] += 1
                    n_blocks += 1
                stall_count = 0

    # Return the best found
    if len(best_blocks) < n_blocks:
        return best_blocks
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k, inv_indptr, inv_indices = _precompute(v, k, t)

    # Multi-start greedy
    best_blocks = None
    best_count = float('inf')
    
    n_starts = 5
    for s in range(n_starts):
        if _elapsed() > _DEADLINE - 40:
            break
        blocks = _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k, seed_offset=s)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on best
    time_budget = _DEADLINE - 8
    blocks = _local_search(best_blocks, cov_matrix, indptr, indices, n_t, n_k, inv_indptr, inv_indices, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    if not np.all(counts > 0):
        while not np.all(counts > 0) and _elapsed() < _DEADLINE - 2:
            missing = np.where(counts == 0)[0]
            if len(missing) == 0: break
            gains = np.sum(cov_matrix[:, missing], axis=1)
            best_i = np.argmax(gains)
            if gains[best_i] == 0: break
            blocks.append(best_i)
            start = indptr[best_i]
            end = indptr[best_i+1]
            if end > start:
                counts[indices[start:end]] += 1

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
