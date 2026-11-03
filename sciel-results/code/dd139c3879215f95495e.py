import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    
    # Map block to list of t-subset indices
    block_to_t_indices = []
    for block in all_k_subsets:
        t_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_t_indices.append(t_indices)
        
    # Map t-subset to list of block indices
    t_to_block_indices = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_t_indices):
        for t_idx in t_indices:
            t_to_block_indices[t_idx].append(b_idx)
            
    # Fast Greedy Initialization
    coverage = np.zeros(num_t_subsets, dtype=np.int8) # 0: uncovered, >0: covered
    block_gain = np.full(num_k_subsets, k, dtype=np.int8) # Wait, gain is number of t-subsets, which is C(k,t)
    # Actually, initial gain is C(k, t)
    initial_gain = len(block_to_t_indices[0])
    block_gain = np.full(num_k_subsets, initial_gain, dtype=np.int16)
    
    selected_blocks = []
    
    def run_greedy():
        nonlocal coverage, block_gain, selected_blocks
        coverage[:] = 0
        block_gain[:] = initial_gain
        selected_blocks = []
        
        remaining = num_t_subsets
        while remaining > 0:
            # Find block with max gain
            max_gain = np.max(block_gain)
            if max_gain == 0:
                break # Should not happen
            
            # Pick one (random tie-breaking for diversity)
            candidates = np.where(block_gain == max_gain)[0]
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_blocks.append(b_idx)
            
            # Update coverage and gains
            for t_idx in block_to_t_indices[b_idx]:
                if coverage[t_idx] == 0:
                    coverage[t_idx] = 1
                    remaining -= 1
                    # Update gains for all blocks containing this t_subset
                    for nb_idx in t_to_block_indices[t_idx]:
                        block_gain[nb_idx] -= 1
                        
    # Run initial greedy
    run_greedy()
    best_blocks = list(selected_blocks)
    best_size = len(best_blocks)
    
    # Local Search / Perturbation
    # We need to be able to add/remove blocks and update state.
    # State: coverage (counts), selected (bool array)
    selected = np.zeros(num_k_subsets, dtype=bool)
    for b in best_blocks:
        selected[b] = True
        
    # Recalculate coverage from scratch for local search state
    coverage_ls = np.zeros(num_t_subsets, dtype=np.int16)
    for b in best_blocks:
        for t_idx in block_to_t_indices[b]:
            coverage_ls[t_idx] += 1
            
    # Helper to try removing a block
    def try_remove(b_idx):
        nonlocal coverage_ls
        # Check if any t_subset becomes uncovered
        can_remove = True
        for t_idx in block_to_t_indices[b_idx]:
            if coverage_ls[t_idx] == 1:
                can_remove = False
                break
        if can_remove:
            for t_idx in block_to_t_indices[b_idx]:
                coverage_ls[t_idx] -= 1
            selected[b_idx] = False
            return True
        return False

    # Helper to repair (add blocks to cover uncovered sets)
    def repair():
        nonlocal coverage_ls, selected
        # Find uncovered
        uncovered = np.where(coverage_ls == 0)[0]
        if len(uncovered) == 0:
            return True
            
        # Calculate gain for each unselected block
        # Gain = number of uncovered t_subsets it covers
        gains = np.zeros(num_k_subsets, dtype=np.int16)
        
        # This is expensive to compute from scratch every time.
        # Optimization: Only compute for blocks that might help.
        # For each uncovered t_idx, update gains of blocks containing it.
        # But we need to reset gains first?
        # Let's just compute gains for all unselected blocks.
        
        # Reset gains for unselected
        # Actually, let's just iterate.
        
        while len(uncovered) > 0:
            # Find best block
            max_gain = 0
            best_b = -1
            for b_idx in range(num_k_subsets):
                if selected[b_idx]:
                    continue
                # Count uncovered covered by b_idx
                g = 0
                for t_idx in block_to_t_indices[b_idx]:
                    if coverage_ls[t_idx] == 0:
                        g += 1
                if g > max_gain:
                    max_gain = g
                    best_b = b_idx
                    
            if best_b == -1 or max_gain == 0:
                return False # Cannot repair
            
            # Add best_b
            selected[best_b] = True
            for t_idx in block_to_t_indices[best_b]:
                if coverage_ls[t_idx] == 0:
                    coverage_ls[t_idx] = 1
                    # Update uncovered list? 
                    # It's easier to just re-check or maintain a set.
                    # For simplicity, we'll just loop until no uncovered.
                    
            # Update uncovered
            uncovered = np.where(coverage_ls == 0)[0]
            
        return True

    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove blocks
        # Random order
        blocks_to_check = list(best_blocks)
        random.shuffle(blocks_to_check)
        for b_idx in blocks_to_check:
            if not selected[b_idx]:
                continue
            if try_remove(b_idx):
                best_blocks.remove(b_idx)
                best_size -= 1
                improved = True
                # Restart check? Or continue?
                # If we removed, the set of blocks to check changes.
                # Just break and restart the loop for safety/simplicity
                break
        
        if improved:
            continue
            
        # Phase 2: Perturbation
        # Pick a random block to remove
        if best_blocks:
            b_to_remove = random.choice(best_blocks)
            
            # Save state
            old_coverage = coverage_ls.copy()
            old_selected = selected.copy()
            old_blocks = list(best_blocks)
            
            # Remove
            for t_idx in block_to_t_indices[b_to_remove]:
                coverage_ls[t_idx] -= 1
            selected[b_to_remove] = False
            best_blocks.remove(b_to_remove)
            
            # Repair
            if repair():
                new_size = sum(selected)
                if new_size < best_size:
                    best_size = new_size
                    # Update best_blocks
                    best_blocks = [i for i in range(num_k_subsets) if selected[i]]
                else:
                    # Revert
                    coverage_ls = old_coverage
                    selected = old_selected
                    best_blocks = old_blocks
            else:
                # Revert
                coverage_ls = old_coverage
                selected = old_selected
                best_blocks = old_blocks

    # Return the best found
    return [list(all_k_subsets[b]) for b in best_blocks]
