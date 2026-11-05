import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    # This is the key structural change from the sparse matrix approach:
    # Direct list access is faster for small k, t
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        # Initial gains: number of subsets covered by each block
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and allow exploration
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            # Find which subsets in this block are currently uncovered
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
                
            # Update uncovered status and gains
            for s in newly_covered:
                uncovered[s] = False
                # Decrement gain for all blocks that cover this subset
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
            
        # Calculate initial gains based only on currently uncovered subsets
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
            
        # Count how many times each subset is covered
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        while True:
            redundant_blocks = []
            
            # Check if removing a block leaves any subset uncovered
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
                
            # Remove a random redundant block
            remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
            b_idx_to_remove = current_blocks[remove_pos]
            
            # Update cover counts
            for s_idx in block_to_subsets[b_idx_to_remove]:
                cover_count[s_idx] -= 1
                
            current_blocks.pop(remove_pos)
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    # Safety margin: 110s out of 120s limit
    deadline = start_time + 110.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        # Only attempt if we have a decent solution and time remains
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 5:
            perturbed = list(best_blocks)
            # Remove a few random blocks to create holes
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask for the remaining set
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair the holes
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine remaining and repaired
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize the new set
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
