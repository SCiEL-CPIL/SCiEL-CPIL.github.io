import numpy as np
import time
import itertools
import random

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
    
    def greedy_construct(uncovered_mask):
        """Run one greedy pass, returning list of blocks."""
        current_blocks = []
        uncovered = uncovered_mask.copy()
        
        # Find uncovered subsets in random order
        uncovered_indices = np.where(uncovered)[0]
        random.shuffle(uncovered_indices)
        
        for idx_s in uncovered_indices:
            if not uncovered[idx_s]:
                continue
            
            s = all_t_subsets[idx_s]
            block = list(s)
            
            # Greedily add points to the block
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
                    
                    gain += random.random() * 0.1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            current_blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
            
            if not uncovered.any():
                break
        
        return current_blocks
    
    def local_search(blocks, time_limit):
        """Aggressive removal and swap local search."""
        blocks = [b[:] for b in blocks]
        search_start = time.time()
        
        # Build coverage counts: for each t-subset, how many blocks cover it
        coverage_count = np.zeros(num_subsets, dtype=np.int32)
        block_subsets = []  # list of lists of subset indices for each block
        
        for block in blocks:
            subs = []
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                coverage_count[idx] += 1
                subs.append(idx)
            block_subsets.append(subs)
        
        improved = True
        while improved and (time.time() - search_start) < time_limit:
            improved = False
            
            # Try to remove blocks that are redundant
            # A block is redundant if every t-subset it covers is covered by at least one other block
            for i in range(len(blocks) - 1, -1, -1):
                if (time.time() - search_start) > time_limit:
                    break
                
                redundant = True
                for idx in block_subsets[i]:
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for idx in block_subsets[i]:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    improved = True
            
            # Try swaps: replace one block with another that covers more uncovered subsets
            if (time.time() - search_start) > time_limit:
                break
            
            # Find subsets covered by exactly one block
            for i in range(len(blocks)):
                if (time.time() - search_start) > time_limit:
                    break
                
                # For each block, try to find a better replacement
                # Score: number of subsets uniquely covered by this block
                unique_count = 0
                for idx in block_subsets[i]:
                    if coverage_count[idx] == 1:
                        unique_count += 1
                
                if unique_count == 0:
                    continue  # Already redundant, will be removed in next pass
                
                # Try random replacements
                for _ in range(5):
                    if (time.time() - search_start) > time_limit:
                        break
                    
                    # Generate a random k-subset
                    new_block = random.sample(range(v), k)
                    new_block.sort()
                    
                    new_subsets = []
                    for sub in itertools.combinations(new_block, t):
                        idx = subset_to_idx[sub]
                        new_subsets.append(idx)
                    
                    # Calculate gain: how many subsets would be uniquely covered by new block
                    # that are not covered by other blocks
                    gain = 0
                    for idx in new_subsets:
                        if coverage_count[idx] == 1:
                            # This subset is currently only covered by some block
                            # If it's covered by block i, removing i would uncover it
                            # But new block covers it, so it stays covered
                            # Actually, we need to check if new block covers subsets
                            # that block i uniquely covers
                            pass
                    
                    # Simpler: compute net change
                    # Subsets lost: those uniquely covered by block i and not by new_block
                    # Subsets gained: those not covered by any other block and covered by new_block
                    new_set = set(new_subsets)
                    old_set = set(block_subsets[i])
                    
                    lost = 0
                    for idx in old_set:
                        if coverage_count[idx] == 1 and idx not in new_set:
                            lost += 1
                    
                    gained = 0
                    for idx in new_set:
                        if coverage_count[idx] == 0:
                            gained += 1
                        elif coverage_count[idx] == 1 and idx in old_set:
                            # Was uniquely covered by i, still covered by new_block
                            # No net change for this subset
                            pass
                    
                    if gained > lost:
                        # Perform swap
                        for idx in block_subsets[i]:
                            coverage_count[idx] -= 1
                        for idx in new_subsets:
                            coverage_count[idx] += 1
                        blocks[i] = new_block
                        block_subsets[i] = new_subsets
                        improved = True
                        break
        
        return blocks
    
    # Run multiple randomized greedy attempts with local search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Greedy construction
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = greedy_construct(uncovered)
        
        # Local search with decreasing time budget
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(remaining * 0.4, 30.0)
            current_blocks = local_search(current_blocks, ls_time)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [b[:] for b in current_blocks]
            
            if best_count <= 131:
                break
    
    if best_blocks is None:
        # Fallback: run one deterministic greedy
        uncovered = np.ones(num_subsets, dtype=bool)
        best_blocks = greedy_construct(uncovered)

    return best_blocks
