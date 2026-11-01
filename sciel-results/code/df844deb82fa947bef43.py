"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: vectorized greedy + count-based local search + k-for-(k+1) exchanges + restarts.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute coverage matrix and index structures."""
    all_t = list(itertools.combinations(range(v), t))
    all_k = list(itertools.combinations(range(v), k))
    n_t = len(all_t)
    n_k = len(all_k)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    # Build coverage matrix: cov_mat[i, j] = 1 if k-subset i covers t-subset j
    cov_mat = np.zeros((n_k, n_t), dtype=np.int8)
    for i, ks in enumerate(all_k):
        for ts in itertools.combinations(ks, t):
            cov_mat[i, t_index[ts]] = 1
    
    # For each t-subset, list of k-subsets that cover it (for swap candidates)
    tsub_to_blocks = [[] for _ in range(n_t)]
    for j in range(n_t):
        tsub_to_blocks[j] = list(np.where(cov_mat[:, j] == 1)[0])
    
    return all_t, all_k, t_index, cov_mat, tsub_to_blocks, n_t, n_k


def _greedy_construct(cov_mat, n_t, n_k, rng, deadline):
    """Vectorized greedy set cover using NumPy."""
    uncovered = np.ones(n_t, dtype=np.int8)
    blocks = []
    
    while uncovered.any() and _elapsed() < deadline:
        gains = cov_mat @ uncovered
        best_i = int(np.argmax(gains))
        if gains[best_i] == 0:
            break
        blocks.append(best_i)
        uncovered = np.maximum(0, uncovered - cov_mat[best_i])
    
    return blocks


def _local_search(blocks, cov_mat, n_t, n_k, tsub_to_blocks, rng, deadline):
    """Count-based local search with removal, swap, and k-for-(k+1) exchanges."""
    blocks = list(blocks)
    
    # Compute cover counts
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        cover_count += cov_mat[b]
    
    improved = True
    while improved and _elapsed() < deadline:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < deadline:
            b = blocks[i]
            # A block is removable if no t-subset it covers has count == 1
            if (cover_count[cov_mat[b] == 1] == 0).all():
                cover_count -= cov_mat[b]
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass (1-for-1) ---
        for _ in range(200):
            if _elapsed() >= deadline:
                break
            i = rng.randrange(len(blocks))
            b_out = blocks[i]
            mask_out = cov_mat[b_out]
            
            # Find t-subsets uniquely covered by b_out
            unique_mask = mask_out & (cover_count == 1)
            if not unique_mask.any():
                # Can just remove it
                cover_count -= mask_out
                blocks.pop(i)
                improved = True
                break
            
            # Find candidates covering all unique t-subsets
            unique_indices = np.where(unique_mask)[0]
            if len(unique_indices) == 0:
                continue
            
            candidates = tsub_to_blocks[int(unique_indices[0])]
            found = False
            for c in rng.sample(candidates, min(len(candidates), 200)):
                if c in blocks:
                    continue
                # Check if c covers all unique t-subsets
                if (cov_mat[c][unique_indices] == 1).all():
                    cover_count -= mask_out
                    cover_count += cov_mat[c]
                    blocks[i] = c
                    found = True
                    improved = True
                    break
            if found:
                break
        
        # --- k-for-(k+1) exchange pass ---
        # Try removing 2 blocks and adding 1 that covers their unique t-subsets
        if len(blocks) >= 3 and _elapsed() < deadline:
            for _ in range(50):
                if _elapsed() >= deadline:
                    break
                i, j = rng.sample(range(len(blocks)), 2)
                b1, b2 = blocks[i], blocks[j]
                mask1, mask2 = cov_mat[b1], cov_mat[b2]
                
                # Unique t-subsets of each (considering both are removed)
                temp_count = cover_count.copy()
                temp_count -= mask1
                temp_count -= mask2
                
                # Check if removing both leaves some uncovered
                if (temp_count < 0).any():
                    continue
                
                # Find t-subsets that need coverage (count was 1 or 2 and now 0 or 1)
                # Actually: t-subsets with temp_count == 0 need a replacement
                need_mask = (temp_count == 0)
                if not need_mask.any():
                    # Both removable together
                    cover_count -= mask1
                    cover_count -= mask2
                    blocks.pop(j)
                    blocks.pop(i)
                    improved = True
                    break
                
                need_indices = np.where(need_mask)[0]
                # Find a block covering all needed t-subsets
                candidates = tsub_to_blocks[int(need_indices[0])]
                found = False
                for c in rng.sample(candidates, min(len(candidates), 200)):
                    if c in blocks:
                        continue
                    if (cov_mat[c][need_indices] == 1).all():
                        cover_count -= mask1
                        cover_count -= mask2
                        cover_count += cov_mat[c]
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(c)
                        found = True
                        improved = True
                        break
                if found:
                    break
        
        # --- 2-for-3 exchange: remove 1, add 2 (net +1, but enables further removal) ---
        # Skip for now to save time
    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)
    
    all_t, all_k, t_index, cov_mat, tsub_to_blocks, n_t, n_k = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different greedy randomizations
    n_restarts = 5
    time_per_restart = (_DEADLINE - 15) / n_restarts
    
    for restart in range(n_restarts):
        t_restart = _DEADLINE - 15 - (n_restarts - 1 - restart) * time_per_restart
        if _elapsed() > t_restart:
            break
        
        # Greedy with slight randomization
        if restart == 0:
            blocks = _greedy_construct(cov_mat, n_t, n_k, rng, t_restart - 5)
        else:
            # Perturbed greedy: randomly skip some candidates
            uncovered = np.ones(n_t, dtype=np.int8)
            blocks = []
            while uncovered.any() and _elapsed() < t_restart - 5:
                gains = cov_mat @ uncovered
                # Add random noise to break ties
                noise = rng.uniform(0, 0.1, size=n_k)
                gains_noisy = gains + noise
                best_i = int(np.argmax(gains_noisy))
                if gains[best_i] == 0:
                    break
                blocks.append(best_i)
                uncovered = np.maximum(0, uncovered - cov_mat[best_i])
        
        # Local search
        blocks = _local_search(blocks, cov_mat, n_t, n_k, tsub_to_blocks, rng, t_restart)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Final polish on best solution
    if best_blocks is not None and _elapsed() < _DEADLINE - 10:
        best_blocks = _local_search(best_blocks, cov_mat, n_t, n_k, tsub_to_blocks, rng, _DEADLINE - 5)
    
    # Safety: verify coverage and repair if needed
    if best_blocks is None:
        # Fallback: pure greedy
        best_blocks = _greedy_construct(cov_mat, n_t, n_k, rng, _DEADLINE - 5)
    
    # Verify and repair
    cov_sum = np.zeros(n_t, dtype=np.int32)
    for b in best_blocks:
        cov_sum += cov_mat[b]
    if (cov_sum == 0).any():
        missing = np.where(cov_sum == 0)[0]
        while len(missing) > 0 and _elapsed() < _DEADLINE - 2:
            gains = cov_mat[:, missing]
            gains_sum = gains.sum(axis=1)
            best_i = int(np.argmax(gains_sum))
            if gains_sum[best_i] == 0:
                break
            best_blocks.append(best_i)
            cov_sum += cov_mat[best_i]
            still_missing = np.where(cov_sum == 0)[0]
            missing = still_missing
    
    # Return as list of k-subsets
    result = [list(all_k[b]) for b in best_blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
