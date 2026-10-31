import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of (t-1)-subsets that contain x
    # Actually, we need: for a partial block B and candidate x, gain = # uncovered t-subsets
    # formed by (t-1)-subset of B plus x.
    
    # Deadline: 105 seconds to be safe (limit is 120, kill is 150)
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy attempts
    while time.time() < deadline:
        # Shuffle the order of subsets to randomize the "first uncovered" choice
        shuffled_subsets = all_t_subsets[:]
        random.shuffle(shuffled_subsets)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        
        for s in shuffled_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
                
            # Build a block containing s
            block = list(s)
            
            # Greedily add points to the block
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain is the number of new t-subsets covered by adding x
                    # These are subsets formed by (t-1) elements from 'block' and x
                    gain = 0
                    # Iterate over all (t-1) subsets of the current block
                    # t=4, so we iterate over 3-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset
                        new_subset = tuple(sorted(list(rest) + [x]))
                        # Check if it's in the mapping and if it's uncovered
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                                
                    # Add small random noise to break ties
                    gain += random.random() * 0.1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.append(best_point)
            
            # Sort the block for consistency (though not strictly required if we use sets)
            block.sort()
            current_blocks.append(block)
            
            # Update uncovered status
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
            
            # Optional: Early termination if all covered? 
            # Not easy to check without scanning, but we can check if uncovered.sum() == 0
            # This is O(N) per block, which is fine.
            if not uncovered.any():
                break
        
        # Check if this run is better
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # If we hit the theoretical lower bound or a very good number, we might want to stop early?
            # Lower bound is 131. If we find 131, we can stop.
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: return a valid covering (should not happen if loop runs at least once)
        # But to be safe, run one deterministic greedy
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        for s in all_t_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
            block = list(s)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(list(rest) + [x]))
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            current_blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
        best_blocks = current_blocks

    # Local Search: ILS with exact intersection repair
    if best_blocks is not None and time.time() < deadline - 10.0:
        best_blocks = local_search(best_blocks, v, k, t, subset_to_idx, deadline)
    
    return best_blocks

def local_search(blocks, v, k, t, subset_to_idx, deadline):
    """
    ILS with exact intersection repair: try to remove blocks that are redundant,
    or swap blocks to maintain validity while reducing size.
    """
    blocks = [b[:] for b in blocks]
    current_count = len(blocks)
    
    # Precompute coverage count for each t-subset
    # coverage_count[i] = number of blocks that cover subset i
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            coverage_count[idx] += 1
    
    # Try to remove redundant blocks (those whose removal doesn't break coverage)
    # We can do this in a random order to escape local minima
    for _ in range(50):  # Max 50 removal attempts
        if time.time() > deadline - 5.0:
            break
        if current_count <= 131:
            break
            
        # Pick a random block to try removing
        idx_to_try = random.randint(0, len(blocks) - 1)
        block_to_remove = blocks[idx_to_try]
        
        # Check if removing this block would leave any t-subset uncovered
        # i.e., for all t-subsets in this block, coverage_count > 1
        safe_to_remove = True
        for sub in itertools.combinations(block_to_remove, t):
            idx = subset_to_idx[sub]
            if coverage_count[idx] <= 1:
                safe_to_remove = False
                break
        
        if safe_to_remove:
            # Remove the block and update coverage counts
            for sub in itertools.combinations(block_to_remove, t):
                idx = subset_to_idx[sub]
                coverage_count[idx] -= 1
            blocks.pop(idx_to_try)
            current_count = len(blocks)
    
    return blocks
