"""Covering design C(19,7,4) using Simulated Annealing."""
import random
import itertools
import numpy as np
from collections import defaultdict
import time

def _precompute_blocks(v, k, t):
    """Precompute all blocks and their t-subset indices."""
    all_points = range(v)
    blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(blocks)
    
    # Map t-subset to index
    t_subs = list(itertools.combinations(all_points, t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices_list = []
    for b in blocks:
        indices = [t_sub_index[ts] for ts in itertools.combinations(b, t)]
        block_t_indices_list.append(indices)
    
    # Convert to numpy array for fast access
    block_t_indices_arr = np.array(block_t_indices_list, dtype=np.int32)
    
    # Precompute for each t-subset, which blocks cover it
    t_sub_to_blocks = defaultdict(list)
    for b_idx, indices in enumerate(block_t_indices_list):
        for t_idx in indices:
            t_sub_to_blocks[t_idx].append(b_idx)
    
    return blocks, block_t_indices_arr, t_sub_to_blocks, n_blocks, n_t_subs

def construct(v, k, t):
    start_time = time.time()
    blocks, block_t_indices, t_sub_to_blocks, n_blocks, n_t_subs = _precompute_blocks(v, k, t)
    
    # State variables
    in_solution = np.zeros(n_blocks, dtype=bool)
    coverage_counts = np.zeros(n_t_subs, dtype=np.int32)
    active_blocks = []
    uncovered_count = n_t_subs
    
    # 1. Greedy Initialization
    uncovered_mask = np.ones(n_t_subs, dtype=bool)
    
    while np.any(uncovered_mask):
        idx = block_t_indices
        vals = uncovered_mask[idx]
        sums = vals.sum(axis=1)
        
        best_block_idx = np.argmax(sums)
        if sums[best_block_idx] == 0:
            break
            
        # Add block
        active_blocks.append(best_block_idx)
        in_solution[best_block_idx] = True
        
        t_idxs = block_t_indices[best_block_idx]
        coverage_counts[t_idxs] += 1
        newly_covered = np.sum(coverage_counts[t_idxs] == 1)
        uncovered_count -= newly_covered
        uncovered_mask[t_idxs] = False
        
        if time.time() - start_time > 5:
            break

    current_size = len(active_blocks)
    current_uncovered = uncovered_count
    
    # Objective: Minimize (uncovered * 1000 + size)
    def get_obj(uncovered, size):
        return uncovered * 1000 + size
        
    current_obj = get_obj(current_uncovered, current_size)
    best_obj = current_obj
    best_blocks = active_blocks.copy()
    best_coverage = coverage_counts.copy()
    best_in_solution = in_solution.copy()
    
    # Simulated Annealing Parameters
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.01
    max_time = 110.0
    
    # Helper to add a block
    def add_block(b_idx):
        nonlocal uncovered_count
        if in_solution[b_idx]:
            return 0
        in_solution[b_idx] = True
        active_blocks.append(b_idx)
        t_idxs = block_t_indices[b_idx]
        old_cov = coverage_counts[t_idxs]
        coverage_counts[t_idxs] += 1
        newly_covered = np.sum((old_cov == 0))
        uncovered_count -= newly_covered
        return -newly_covered * 1000 + 1 # Delta obj
        
    # Helper to remove a block
    def remove_block(b_idx):
        nonlocal uncovered_count
        if not in_solution[b_idx]:
            return 0
        in_solution[b_idx] = False
        active_blocks.remove(b_idx)
        t_idxs = block_t_indices[b_idx]
        old_cov = coverage_counts[t_idxs]
        coverage_counts[t_idxs] -= 1
        newly_uncovered = np.sum((old_cov == 1))
        uncovered_count += newly_uncovered
        return newly_uncovered * 1000 - 1 # Delta obj

    while time.time() - start_time < max_time and T > min_T:
        
        # Generate move
        move_type = random.randint(0, 2)
        delta_obj = 0
        
        if move_type == 0 and current_size < n_blocks:
            # Add random block
            b_idx = random.randint(0, n_blocks - 1)
            if not in_solution[b_idx]:
                delta_obj = add_block(b_idx)
                # Check acceptance
                if random.random() < np.exp(-delta_obj / T):
                    # Accept
                    pass
                else:
                    # Reject: Undo
                    remove_block(b_idx)
        elif move_type == 1 and current_size > 0:
            # Remove random block
            b_idx = random.choice(active_blocks)
            delta_obj = remove_block(b_idx)
            if random.random() < np.exp(-delta_obj / T):
                pass
            else:
                # Reject: Undo
                add_block(b_idx)
        elif move_type == 2 and current_size > 0:
            # Swap
            b_idx_in = random.choice(active_blocks)
            b_idx_out = random.randint(0, n_blocks - 1)
            if not in_solution[b_idx_out] and b_idx_in != b_idx_out:
                d1 = remove_block(b_idx_in)
                d2 = add_block(b_idx_out)
                delta_obj = d1 + d2
                if random.random() < np.exp(-delta_obj / T):
                    pass
                else:
                    # Reject: Undo
                    remove_block(b_idx_out)
                    add_block(b_idx_in)
        
        current_obj = get_obj(uncovered_count, current_size)
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_blocks = active_blocks.copy()
            best_coverage = coverage_counts.copy()
            best_in_solution = in_solution.copy()
            
        T *= cooling_rate

    # Final Local Search to polish
    # Try to remove any redundant blocks
    for b_idx in best_blocks[:]:
        if not best_in_solution[b_idx]:
            continue
        t_idxs = block_t_indices[b_idx]
        if np.any(best_coverage[t_idxs] == 1):
            continue
        # Remove
        best_in_solution[b_idx] = False
        best_coverage[t_idxs] -= 1
        best_blocks.remove(b_idx)
        
    final_blocks = [list(blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
