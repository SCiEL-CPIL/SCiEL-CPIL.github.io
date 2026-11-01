"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: greedy construction with randomization + aggressive local search (deletion, swap, add).
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and coverage structures."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # coverage_arr: 2D numpy array [n_k, n_t] with 0/1
    cov_rows = []
    for ks in k_subsets:
        row = np.zeros(n_t, dtype=np.int8)
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            row[idx] = 1
        cov_rows.append(row)
    
    coverage_arr = np.array(cov_rows, dtype=np.int8)
    
    # tsub_to_blocks: for each t-subset, list of k-subset indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t):
            if coverage_arr[i, j]:
                tsub_to_blocks[j].append(i)
    
    return all_t, t_index, k_subsets, coverage_arr, tsub_to_blocks


def _greedy_construct(v, k, t, rng, coverage_arr, tsub_to_blocks):
    """Greedy set cover with randomization."""
    n_t = len(coverage_arr[0])
    n_k = len(coverage_arr)
    uncovered = np.ones(n_t, dtype=np.int8)
    blocks = []
    
    while np.any(uncovered) and _elapsed() < _DEADLINE - 30:
        # Compute gain for each candidate block
        gains = coverage_arr @ uncovered
        best_gain = int(np.max(gains))
        
        if best_gain <= 0:
            break
        
        # Collect all blocks with max gain
        max_idx = np.where(gains == best_gain)[0]
        
        # Randomly select from top candidates to diversify
        if len(max_idx) > 1:
            # Pick from top few
            top_n = min(5, len(max_idx))
            chosen = rng.choice(max_idx[:top_n])
        else:
            chosen = max_idx[0]
        
        blocks.append(int(chosen))
        uncovered = np.maximum(0, uncovered - coverage_arr[chosen])
        
    return blocks


def _local_search(blocks, coverage_arr, tsub_to_blocks, rng, time_budget):
    """Aggressive local search: deletion, swap, add passes."""
    n_t = len(coverage_arr[0])
    n_k = len(coverage_arr)
    blocks = list(blocks)
    
    # Compute cover count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        cover_count += coverage_arr[b]
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Deletion pass: remove blocks that don't uniquely cover anything ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            row = coverage_arr[b]
            # Check if this block uniquely covers any t-subset
            if np.any((row == 1) & (cover_count == 1)):
                i += 1
            else:
                cover_count -= row
                blocks.pop(i)
                improved = True
        
        # --- Check validity and repair if needed ---
        if np.any(cover_count == 0):
            # Add blocks to cover uncovered t-subsets
            while np.any(cover_count == 0) and _elapsed() < time_budget:
                missing = (cover_count == 0).astype(np.int8)
                gains = coverage_arr @ missing
                best_i = int(np.argmax(gains))
                if gains[best_i] == 0:
                    break
                blocks.append(best_i)
                cover_count += coverage_arr[best_i]
                improved = True
        
        # --- Swap pass: try to replace blocks with better ones ---
        n_swaps = 50
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            
            if len(blocks) == 0:
                break
                
            i = rng.randrange(len(blocks))
            b_out = blocks[i]
            row_out = coverage_arr[b_out]
            
            # Find t-subsets uniquely covered by this block
            unique_mask = (row_out == 1) & (cover_count == 1)
            if not np.any(unique_mask):
                # Can remove this block
                cover_count -= row_out
                blocks.pop(i)
                improved = True
                break
            
            unique_indices = np.where(unique_mask)[0]
            
            # Try to find a replacement that covers all unique t-subsets
            # Use the first unique t-subset to find candidates
            first_idx = unique_indices[0]
            candidates = tsub_to_blocks[first_idx]
            
            found = False
            for c in candidates:
                if c in blocks:
                    continue
                row_c = coverage_arr[c]
                # Check if this candidate covers all unique t-subsets
                if np.all(row_c[unique_indices] == 1):
                    # Check if swap reduces total coverage waste
                    old_waste = np.sum(cover_count > 1)
                    new_cover = cover_count - row_out + row_c
                    new_waste = np.sum(new_cover > 1)
                    
                    if new_waste <= old_waste:
                        cover_count = new_cover
                        blocks[i] = c
                        found = True
                        improved = True
                        break
            
            if not found:
                continue
    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)

    all_t, t_index, k_subsets, coverage_arr, tsub_to_blocks = _precompute(v, k, t)
    n_t = len(all_t)
    n_k = len(k_subsets)

    best_blocks = None
    best_len = float('inf')
    
    # Try multiple random seeds for greedy construction
    n_attempts = 5
    for attempt in range(n_attempts):
        if _elapsed() > _DEADLINE - 40:
            break
            
        seed = 42 + attempt * 1000
        rng_attempt = random.Random(seed)
        
        # Greedy construction
        blocks = _greedy_construct(v, k, t, rng_attempt, coverage_arr, tsub_to_blocks)
        
        # Local search
        time_budget = _DEADLINE - 15 - (n_attempts - attempt - 1) * 5
        blocks = _local_search(blocks, coverage_arr, tsub_to_blocks, rng_attempt, time_budget)
        
        # Deduplicate
        seen = set()
        dedup = []
        for b in blocks:
            key = tuple(k_subsets[b])
            if key not in seen:
                seen.add(key)
                dedup.append(b)
        
        if len(dedup) < best_len:
            best_len = len(dedup)
            best_blocks = dedup
    
    # Final safety check and repair
    if best_blocks is None:
        # Fallback: pure greedy
        best_blocks = _greedy_construct(v, k, t, rng, coverage_arr, tsub_to_blocks)
    
    # Verify coverage and repair if needed
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in best_blocks:
        cover_count += coverage_arr[b]
    
    if np.any(cover_count == 0):
        while np.any(cover_count == 0) and _elapsed() < _DEADLINE - 2:
            missing = (cover_count == 0).astype(np.int8)
            gains = coverage_arr @ missing
            best_i = int(np.argmax(gains))
            if gains[best_i] == 0:
                break
            best_blocks.append(best_i)
            cover_count += coverage_arr[best_i]
    
    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in best_blocks]
    # Final deduplication
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
