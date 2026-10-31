"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + aggressive local search (removal/swap/add).
"""
import time
import itertools
import random

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets as ints, and for every k-subset its coverage bitmap."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    # For each k-subset, the set of t-subset indices it covers
    k_subsets = list(itertools.combinations(range(v), k))
    # coverage: list of ints (bitmasks) for each k-subset
    coverage_bits = []
    for ks in k_subsets:
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
        coverage_bits.append(mask)
    return all_t, t_index, k_subsets, coverage_bits


def _greedy_construct(v, k, t, rng):
    """Greedy set cover: repeatedly add the k-subset covering the most uncovered t-subsets."""
    all_t, t_index, k_subsets, coverage_bits = _precompute(v, k, t)
    n_t = len(all_t)
    # uncovered as a bitmask
    uncovered = (1 << n_t) - 1
    blocks = []  # list of k-subset index
    
    # Optimization: Precompute list of (index, mask)
    # We need to find max popcount(coverage[i] & uncovered)
    # Using Python's int.bit_count() is fast.
    
    while uncovered and _elapsed() < _DEADLINE - 20:
        best_i = -1
        best_gain = 0
        # Iterate through all k-subsets
        # 50388 iterations * bit_count cost. 
        # bit_count on 3876-bit int is very fast.
        for i, mask in enumerate(coverage_bits):
            g = (mask & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
                if best_gain == 0:
                    break
        if best_i < 0:
            break
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
    return blocks, all_t, t_index, k_subsets, coverage_bits


def _local_search(blocks, coverage_bits, n_t, k_subsets, rng, time_budget):
    """Removal + swap + add local search to minimize |blocks| while staying a cover."""
    blocks = list(blocks)
    
    # Helper to check if removing block i keeps coverage
    # We need union of all blocks except i to be full mask.
    # full_mask = (1 << n_t) - 1
    
    full_mask = (1 << n_t) - 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Check if removing b still covers everything
            # Compute union of all blocks except i
            cov = 0
            for idx, bb in enumerate(blocks):
                if idx != i:
                    cov |= coverage_bits[bb]
            
            if cov == full_mask:
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass ---
        # Try swapping a block with another to enable removal
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(len(k_subsets))
            if j in blocks:
                continue
            
            old = blocks[i]
            blocks[i] = j
            
            # Check if we can remove any block now
            removed = False
            for idx in range(len(blocks)):
                cov = 0
                for idx2 in range(len(blocks)):
                    if idx2 != idx:
                        cov |= coverage_bits[blocks[idx2]]
                
                if cov == full_mask:
                    blocks.pop(idx)
                    removed = True
                    break
            
            if removed:
                improved = True
            else:
                blocks[i] = old  # revert
        
        # --- Add pass: if not covered (shouldn't happen), add best block ---
        if len(blocks) > 0:
            cov = 0
            for b in blocks:
                cov |= coverage_bits[b]
            
            if cov != full_mask:
                missing = full_mask ^ cov
                best_i = -1
                best_g = 0
                for idx, mask in enumerate(coverage_bits):
                    g = (mask & missing).bit_count()
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    blocks.append(best_i)
                    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage_bits = _precompute(v, k, t)
    n_t = len(all_t)

    # Greedy construction
    blocks, _, _, _, _ = _greedy_construct(v, k, t, rng)
    
    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, coverage_bits, n_t, k_subsets, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    cov = 0
    for b in blocks:
        cov |= coverage_bits[b]
        
    full_mask = (1 << n_t) - 1
    if cov != full_mask:
        # repair with greedy add
        while cov != full_mask and _elapsed() < _DEADLINE - 2:
            missing = full_mask ^ cov
            best_i = -1
            best_g = 0
            for idx, mask in enumerate(coverage_bits):
                g = (mask & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage_bits[best_i]

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
