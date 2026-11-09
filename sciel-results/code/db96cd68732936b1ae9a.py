"""Improved program for C(19, 7, 4) covering design."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    seed = 0
    
    while time.time() < deadline:
        blocks = greedy_cover(v, k, t, all_t_subsets, subset_to_idx, num_t_subsets, seed)
        blocks = remove_redundant(blocks, t, subset_to_idx)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            # Lower bound is 131. If we hit it, we can stop.
            if len(blocks) <= 131:
                break
        
        seed += 1
        
    return best_blocks if best_blocks is not None else []


def greedy_cover(v, k, t, all_t_subsets, subset_to_idx, num_t_subsets, seed):
    random.seed(seed)
    # Bitset for uncovered subsets
    # Initialize with all bits set
    uncovered = (1 << num_t_subsets) - 1
    blocks = []
    
    # Create a shuffled order of indices to pick start subsets
    # This adds randomness to which uncovered subset we pick first
    perm = list(range(num_t_subsets))
    random.shuffle(perm)
    
    while uncovered:
        # Find the first uncovered subset according to the perm order
        start_idx = -1
        for idx in perm:
            if (uncovered >> idx) & 1:
                start_idx = idx
                break
        
        if start_idx == -1:
            break # Should not happen if uncovered is non-zero
            
        start_subset = all_t_subsets[start_idx]
        block = list(start_subset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            
            block_set = set(block)
            
            # Pre-compute combinations of block of size t-1
            # To avoid recomputing for every x, we can just iterate
            # But we need to check gain for each x
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check subsets T = {x} U C where C is (t-1)-subset of block
                # T must be in uncovered
                for combo in itertools.combinations(block, t - 1):
                    # Create sorted tuple
                    # combo is a tuple of t-1 elements from block
                    # x is an int
                    # We need to merge and sort
                    # Since t is small (4), sorting is fast
                    cand_tuple = tuple(sorted(combo + (x,)))
                    # Look up index
                    # It must exist because it's a valid t-subset
                    c_idx = subset_to_idx[cand_tuple]
                    if (uncovered >> c_idx) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and gain > 0:
                    # Tie breaking: random
                    if random.random() < 0.5:
                        best_point = x
                        best_gain = gain
                        # We can break early if we want, but checking all allows better random tie-breaking
                        # Or just pick one. Let's pick one to save time.
                        # Actually, just keeping the first one found with max gain is fine.
                        # The random.shuffle of perm handles diversity.
                        # So we don't need random tie-breaking here.
                        pass
            
            if best_point == -1:
                # No gain found? Should not happen if uncovered is not empty and k > t
                # Pick a random point not in block
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_point = random.choice(available)
                else:
                    break
            
            # Insert best_point into block maintaining sorted order
            block.append(best_point)
            block.sort()
        
        blocks.append(block)
        
        # Update uncovered bitset
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            uncovered &= ~(1 << idx)
            
    return blocks


def remove_redundant(blocks, t, subset_to_idx):
    if not blocks:
        return blocks
    
    # Count coverage for each t-subset
    # Using a defaultdict for simplicity
    counts = defaultdict(int)
    
    for block in blocks:
        for sub in itertools.combinations(block, t):
            counts[sub] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            # Check if this block is redundant
            # A block is redundant if for all sub in C(block, t), counts[sub] > 1
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if counts[sub] <= 1:
                    is_redundant = False
                    break
                
            if is_redundant:
                # Remove this block
                for sub in itertools.combinations(block, t):
                    counts[sub] -= 1
                changed = True
            else:
                new_blocks.append(block)
                
        blocks = new_blocks
        
    return blocks
# EVOLVE-BLOCK-END
