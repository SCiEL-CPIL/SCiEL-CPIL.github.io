import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    # block_to_subsets is a list of lists, but we'll keep it as is for now
    # For faster gain updates, we can precompute the sizes
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and explore
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            # Find newly covered subsets
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                for b in subset_to_blocks[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_to_blocks[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if current_uncovered[s]]
            
            if not newly_covered:
                continue
                
            for s in newly_covered:
                current_uncovered[s] = False
                for b in subset_to_blocks[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        # Repeatedly remove redundant blocks
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                is_redundant = True
                for s_idx in block_to_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            # Remove all redundant blocks at once for efficiency
            for remove_pos in sorted(redundant_blocks, reverse=True):
                b_idx_to_remove = current_blocks[remove_pos]
                
                for s_idx in block_to_subsets[b_idx_to_remove]:
                    cover_count[s_idx] -= 1
                    
                current_blocks.pop(remove_pos)
                changed = True
            
        return current_blocks

    def local_search_swap(blocks_indices, random_seed=None, max_iters=100):
        """Try to swap blocks to reduce the total number."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if len(current_blocks) < 2:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
        
        # Try to find a block that can be removed and replaced by a better one
        # Or find two blocks where one can be swapped for a different one
        # For now, let's just try removing one block and repairing
        best = list(current_blocks)
        
        for _ in range(max_iters):
            if not current_blocks:
                break
            # Pick a random block to remove
            remove_idx = rng.randint(0, len(current_blocks) - 1)
            b_remove = current_blocks[remove_idx]
            
            # Temporarily remove it
            new_cover = cover_count.copy()
            for s_idx in block_to_subsets[b_remove]:
                new_cover[s_idx] -= 1
            
            uncovered_mask = (new_cover == 0).astype(bool)
            
            if not np.any(uncovered_mask):
                # This block was redundant
                current_blocks.pop(remove_idx)
                cover_count = new_cover
                continue
            
            # Repair
            repair = run_greedy_partial(uncovered_mask, random_seed + _ + 10000)
            
            if len(repair) == 0:
                # Should not happen since we have uncovered
                continue
                
            # If repair adds fewer than 1 block, we made progress
            # But repair adds at least 1 block, so net change is 0
            # Unless repair adds 0 blocks (impossible)
            # So this swap strategy doesn't help directly.
            # Let's try a different approach: try to replace a block with a different one
            # that covers the same uncovered subsets but also covers some additional ones
            # that are currently covered by another block, allowing that other block to be removed.
            
            # For now, let's just try to find a block that can be swapped
            # Pick a random block to swap
            swap_idx = rng.randint(0, len(current_blocks) - 1)
            b_swap = current_blocks[swap_idx]
            
            # Find all blocks that cover the same subsets as b_swap
            # This is expensive, so let's just try random blocks
            for _ in range(10):
                new_b = rng.randint(0, num_blocks - 1)
                if new_b == b_swap:
                    continue
                if new_b in current_blocks:
                    continue
                    
                # Check if swapping b_swap with new_b maintains coverage
                # Remove b_swap, add new_b
                test_cover = cover_count.copy()
                for s_idx in block_to_subsets[b_swap]:
                    test_cover[s_idx] -= 1
                for s_idx in block_to_subsets[new_b]:
                    test_cover[s_idx] += 1
                    
                if np.all(test_cover > 0):
                    # Swap is valid
                    current_blocks[swap_idx] = new_b
                    cover_count = test_cover
                    # Now try to remove redundant blocks
                    redundant = []
                    for i, b_idx in enumerate(current_blocks):
                        is_redundant = True
                        for s_idx in block_to_subsets[b_idx]:
                            if cover_count[s_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            redundant.append(i)
                    for remove_pos in sorted(redundant, reverse=True):
                        b_to_remove = current_blocks[remove_pos]
                        for s_idx in block_to_subsets[b_to_remove]:
                            cover_count[s_idx] -= 1
                        current_blocks.pop(remove_pos)
                    break
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Local Search on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 5:
            ls_res = local_search_swap(best_blocks, trial + 5000, max_iters=20)
            min_res_ls = minimize(ls_res, trial + 6000)
            
            if len(min_res_ls) < best_len:
                best_len = len(min_res_ls)
                best_blocks = min_res_ls
        
        # Strategy 3: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 5:
            perturbed = list(best_blocks)
            # Remove 2-3 random blocks
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
