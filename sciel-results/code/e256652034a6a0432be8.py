"""Improved greedy set-cover for C(19,8,5) with efficient incremental gain updates."""
import time
import itertools
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_to_idx = {tuple(c): i for i, c in enumerate(all_t)}

    # For each t-subset, precompute which k-subsets (blocks) contain it
    # A block contains a t-subset S if S is a subset of the block
    # We'll use a different approach: for each t-subset, we know which blocks cover it
    
    # Actually, let's use the standard greedy set cover approach:
    # 1. Generate all possible k-subsets as potential blocks
    # 2. For each block, compute which t-subsets it covers
    # 3. Greedily select blocks with maximum marginal coverage
    
    # But C(19,8) = 75582 blocks, each covering C(8,5) = 56 t-subsets
    # Total t-subsets: C(19,5) = 11628
    
    # Precompute: for each possible block (k-subset), which t-subsets does it cover?
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    
    # Map t-subset index to list of block indices that cover it
    blocks_covering_t = defaultdict(list)
    
    # For each block, find all t-subsets it covers
    block_t_coverage = []  # list of lists of t-subset indices
    for blk in all_k:
        covered_t = []
        for S in itertools.combinations(blk, t):
            idx = t_to_idx.get(tuple(S))
            if idx is not None:
                covered_t.append(idx)
                blocks_covering_t[idx].append(len(block_t_coverage))
        block_t_coverage.append(covered_t)
    
    # Greedy set cover with incremental gain updates
    covered = np.zeros(n_t, dtype=bool)
    remaining = n_t
    gain = np.array([len(ct) for ct in block_t_coverage], dtype=np.int32)
    used = np.zeros(n_k, dtype=bool)
    blocks = []
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find block with maximum gain
        # Mask out used blocks
        masked_gain = gain.copy()
        masked_gain[used] = -1
        b_star = int(np.argmax(masked_gain))
        
        if masked_gain[b_star] <= 0:
            break
        
        # Add this block
        used[b_star] = True
        blocks.append(list(all_k[b_star]))
        
        # Update coverage and gains
        for t_idx in block_t_coverage[b_star]:
            if not covered[t_idx]:
                covered[t_idx] = True
                remaining -= 1
                # Decrease gain for all blocks that also cover this t-subset
                for b in blocks_covering_t[t_idx]:
                    if not used[b]:
                        gain[b] -= 1
    
    return blocks
