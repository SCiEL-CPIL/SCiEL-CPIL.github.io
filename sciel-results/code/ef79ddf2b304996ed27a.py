"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses optimized greedy with multiple random restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {}
    for i, combo in enumerate(all_t_subsets):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        t_subset_to_idx[mask] = i
    
    # For each point x and each (t-1)-subset S of points not containing x,
    # precompute which t-subset index contains x and S
    # This helps in fast gain computation
    # gain_table[x][S_mask] = t_subset_idx or -1
    gain_table = [[-1] * (1 << v) for _ in range(v)]
    for i, combo in enumerate(all_t_subsets):
        for j in range(t):
            x = combo[j]
            rest = combo[:j] + combo[j+1:]
            rest_mask = 0
            for y in rest:
                rest_mask |= (1 << y)
            gain_table[x][rest_mask] = i
    
    def greedy_cover(seed):
        """Run a greedy covering algorithm with a given random seed."""
        random.seed(seed)
        
        # uncovered: boolean array for fast membership test
        uncovered = [True] * n_t
        uncovered_count = n_t
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset to start the block
            # Find an uncovered index
            start_idx = -1
            for i in range(n_t):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break
            
            start_combo = all_t_subsets[start_idx]
            block = list(start_combo)
            random.shuffle(block)
            
            while len(block) < k:
                # For each candidate point, compute how many uncovered t-subsets it would cover
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets contain x and the other t-1 points from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_mask = 0
                        for y in rest:
                            rest_mask |= (1 << y)
                        idx = gain_table[x][rest_mask]
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    # Fallback: pick a random point not in block
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = random.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for combo in itertools.combinations(block, t):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                idx = t_subset_to_idx[mask]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        return blocks
    
    def try_remove_blocks(blocks, deadline):
        """Try to remove blocks that are redundant."""
        # Precompute t-subsets for each block as masks
        block_t_masks = []
        for block in blocks:
            masks = []
            for combo in itertools.combinations(block, t):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                masks.append(mask)
            block_t_masks.append(masks)
        
        # Count coverage for each t-subset
        coverage_count = {}
        for masks in block_t_masks:
            for mask in masks:
                coverage_count[mask] = coverage_count.get(mask, 0) + 1
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2:
                break
            for i in range(len(blocks)):
                if time.time() > deadline - 2:
                    break
                # Check if block i is redundant
                redundant = True
                for mask in block_t_masks[i]:
                    if coverage_count.get(mask, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for mask in block_t_masks[i]:
                        coverage_count[mask] -= 1
                    blocks.pop(i)
                    block_t_masks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    # Try multiple random restarts
    best_blocks = None
    best_len = float('inf')
    
    # First, try a deterministic greedy (seed=0)
    blocks = greedy_cover(0)
    blocks = try_remove_blocks(blocks, deadline)
    best_blocks = blocks
    best_len = len(blocks)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_blocks
    
    # Try more random restarts
    for seed in range(1, 500):
        if time.time() > deadline - 5:
            break
        blocks = greedy_cover(seed)
        blocks = try_remove_blocks(blocks, deadline)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    return best_blocks

# EVOLVE-BLOCK-END
