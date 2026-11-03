"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
        
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_id_to_indices: list of numpy arrays
    block_id_to_indices = [None] * n_blocks
    # subset_to_block_ids: list of lists of block ids
    subset_to_block_ids = [[] for _ in range(n_subsets)]
    
    for i, block in enumerate(all_blocks):
        idxs = []
        for c in itertools.combinations(block, t):
            idx = subset_to_idx[c]
            idxs.append(idx)
            subset_to_block_ids[idx].append(i)
        block_id_to_indices[i] = np.array(idxs, dtype=np.int32)
        
    data = (all_blocks, block_id_to_indices, subset_to_block_ids, n_subsets)
    _CACHE[(v, k, t)] = data
    return data

def _greedy_cover(v, k, t, rng, all_blocks, block_id_to_indices, subset_to_block_ids, n_subsets):
    counts = np.zeros(n_subsets, dtype=np.int32)
    uncovered = np.ones(n_subsets, dtype=bool)
    selected_block_ids = []
    
    while np.any(uncovered):
        # Find an uncovered subset
        # Using np.nonzero is O(n). n=2024. Fast.
        # But doing it every step (76 steps) is fine.
        # Or random sampling.
        # Random sampling is O(1) expected if density is high.
        # Initially density is 1.0.
        
        while True:
            idx = rng.randint(0, n_subsets - 1)
            if uncovered[idx]:
                break
        
        # Pick a random block covering this subset
        candidates = subset_to_block_ids[idx]
        bid = rng.choice(candidates)
        
        selected_block_ids.append(bid)
        
        # Update counts
        indices = block_id_to_indices[bid]
        counts[indices] += 1
        # Update uncovered
        uncovered[indices] = (counts[indices] == 0)
        
    return selected_block_ids

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    rng = random.Random(12345)
    
    all_blocks, block_id_to_indices, subset_to_block_ids, n_subsets = _get_precomputed(v, k, t)
    
    # Initial greedy
    selected_ids = _greedy_cover(v, k, t, rng, all_blocks, block_id_to_indices, subset_to_block_ids, n_subsets)
    
    counts = np.zeros(n_subsets, dtype=np.int32)
    for bid in selected_ids:
        indices = block_id_to_indices[bid]
        counts[indices] += 1
        
    best_count = len(selected_ids)
    best_ids = list(selected_ids)
    
    # Local Search
    # We maintain a set of selected block IDs for O(1) removal? 
    # Or just a list.
    # List is fine for N=76.
    
    # Optimization: Keep track of which blocks are selected?
    # For swap, we pick a random block from the universe.
    # We need to ensure we don't swap to a block already in the solution (unless we allow duplicates, but task says distinct blocks).
    # "blocks must be distinct"
    
    while time.time() - start_time < time_limit:
        n_sel = len(selected_ids)
        
        # 1. Try Removal
        if n_sel > 0:
            pos = rng.randint(0, n_sel - 1)
            bid = selected_ids[pos]
            indices = block_id_to_indices[bid]
            if np.all(counts[indices] > 1):
                # Remove
                counts[indices] -= 1
                selected_ids.pop(pos)
                if len(selected_ids) < best_count:
                    best_count = len(selected_ids)
                    best_ids = list(selected_ids)
                continue # Restart loop to try more removals
        
        # 2. Try Swap
        if n_sel > 0:
            pos = rng.randint(0, n_sel - 1)
            old_bid = selected_ids[pos]
            old_indices = block_id_to_indices[old_bid]
            
            # Pick new block
            new_bid = rng.randint(0, len(all_blocks) - 1)
            if new_bid == old_bid:
                continue
            # Check if new_bid is already in selected_ids
            # O(N) check. N=76. Fine.
            if new_bid in selected_ids:
                continue
                
            new_indices = block_id_to_indices[new_bid]
            
            # Check validity:
            # Any subset covered only by old_bid and NOT by new_bid must have count > 1 (so it remains covered by others)
            # i.e., for all i in old_indices:
            #   if i not in new_indices: counts[i] must be > 1
            #   if i in new_indices: counts[i] can be 1 (it will become 2 then 1, or 1 then 2? No, we remove old then add new)
            # Actually, we can check:
            #   subsets in old_indices \ new_indices must have counts > 1.
            
            # Efficient check:
            # Create a mask for old_indices?
            # Or just loop? 35 elements.
            
            valid = True
            for idx in old_indices:
                if counts[idx] == 1:
                    # This subset is ONLY covered by old_bid.
                    # It MUST be covered by new_bid.
                    if idx not in new_indices:
                        valid = False
                        break
                        
            if valid:
                # Apply swap
                counts[old_indices] -= 1
                counts[new_indices] += 1
                selected_ids[pos] = new_bid
                
                if len(selected_ids) < best_count:
                    best_count = len(selected_ids)
                    best_ids = list(selected_ids)
                    
    return [list(all_blocks[bid]) for bid in best_ids]
# EVOLVE-BLOCK-END
