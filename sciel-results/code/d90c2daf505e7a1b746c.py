import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precomputation
    points = range(v)
    all_t_subs = list(itertools.combinations(points, t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_subs = len(all_t_subs)
    
    all_k_blocks = list(itertools.combinations(points, k))
    block_to_subs = {}
    for b in all_k_blocks:
        subs = [sub_to_idx[tuple(s)] for s in itertools.combinations(b, t)]
        block_to_subs[b] = subs
        
    # Greedy Init
    coverage = np.zeros(n_subs, dtype=np.int32)
    uncovered_mask = np.ones(n_subs, dtype=bool)
    blocks = []
    remaining_uncovered = n_subs
    
    # Precompute for speed in greedy: for each block, list of sub indices
    # We already have block_to_subs
    
    # Greedy loop
    # To make greedy fast, we can iterate through blocks and find the one covering most uncovered.
    # But 50k blocks * 160 blocks * 35 subs is ~280M ops. Might be too slow in Python.
    # Optimization: Only consider blocks that cover at least one uncovered subset?
    # Or use a faster heuristic for the first few blocks, then switch to SA.
    
    # Let's try a faster greedy:
    # Randomize block order? No, greedy picks best.
    # We can approximate gain.
    
    # Actually, let's just run SA from a random valid solution or a quick greedy.
    # Quick Greedy:
    while remaining_uncovered > 0:
        best_block = None
        best_gain = -1
        # Sample a subset of blocks to speed up greedy? Or just loop all?
        # 50388 blocks. 35 subs each. 1.7M checks per block selection.
        # If 150 blocks, 250M checks. Too slow for Python.
        
        # Alternative: Pick a random uncovered subset, then pick a random block containing it?
        # This is "Greedy on Subsets".
        # Pick random uncovered subset S.
        # Find all blocks containing S.
        # Pick the one that covers the most *other* uncovered subsets.
        
        # Let's implement this.
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0: break
        rand_sub_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
        rand_sub = all_t_subs[rand_sub_idx]
        
        # Find blocks containing rand_sub
        # This requires an index: subset -> list of blocks containing it.
        # Precompute this.
        
    # This precomputation (subset -> blocks) is memory heavy but 3876 * ~50 blocks = 200k entries. Fine.
    
    # Let's restart with this plan.
