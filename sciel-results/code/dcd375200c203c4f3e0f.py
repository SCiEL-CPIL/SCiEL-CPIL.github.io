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
    
    # coverage_counts: how many blocks in current solution cover each t-subset
    coverage_counts = np.zeros(n_t_subs, dtype=np.int32)
    # in_solution: boolean mask for blocks
    in_solution = np.zeros(n_blocks, dtype=bool)
    
    solution_blocks = [] # List of block indices
    
    # 1. Greedy Initialization
    uncovered_mask = np.ones(n_t_subs, dtype=bool)
    
    while np.any(uncovered_mask):
        # Vectorized coverage count for all blocks
        idx = block_t_indices
        vals = uncovered_mask[idx]
        sums = vals.sum(axis=1)
        
        best_block_idx = np.argmax(sums)
        if sums[best_block_idx] == 0:
            break
            
        # Add block
        solution_blocks.append(best_block_idx)
        in_solution[best_block_idx] = True
        
        # Update coverage counts and uncovered mask
        t_idxs = block_t_indices[best_block_idx]
        coverage_counts[t_idxs] += 1
        uncovered_mask[t_idxs] = False
        
        if time.time() - start_time > 5:
            break

    # 2. Simulated Annealing
    current_len = len(solution_blocks)
    best_len = current_len
    best_solution = solution_blocks.copy()
    
    # Energy function: number of uncovered t-subsets
    # We want to minimize this. Ideally 0.
    # But we also want to minimize the number of blocks.
    # We can define energy = current_len + lambda * uncovered_count
    # Or just focus on minimizing len, and use uncovered_count as a penalty for invalid moves.
    # Standard SA for Set Cover: 
    # Move: Remove/Replace block.
    # Delta Energy: Change in (num_blocks + penalty * num_uncovered)
    # Let's use: Cost = num_blocks + 100 * num_uncovered
    # This ensures validity is prioritized, then minimization.
    
    penalty = 100
    current_cost = current_len + penalty * int(np.sum(coverage_counts == 0))
    best_cost = current_cost
    
    T = 10.0 # Initial temperature
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < 110:
        if T < T_min:
            break
            
        # Choose a random move
        if random.random() < 0.5 and len(solution_blocks) > 0:
            # Try to remove a random block
            idx_in_sol = random.randint(0, len(solution_blocks) - 1)
            b_idx = solution_blocks[idx_in_sol]
            
            t_idxs = block_t_indices[b_idx]
            # Check if any t-subset becomes uncovered
            new_uncovered = np.sum(coverage_counts[t_idxs] == 1)
            
            # Delta cost:
            # Removing block decreases num_blocks by 1.
            # If new_uncovered > 0, it increases uncovered count by new_uncovered.
            delta_cost = -1 + penalty * new_uncovered
            
            if delta_cost <= 0 or (random.random() < np.exp(-delta_cost / T)):
                # Accept move
                solution_blocks.pop(idx_in_sol)
                in_solution[b_idx] = False
                coverage_counts[t_idxs] -= 1
                
                current_len -= 1
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_len = current_len
                    best_solution = solution_blocks.copy()
                    
        else:
            # Try to swap a random block with another
            if len(solution_blocks) == 0:
                continue
                
            idx_in_sol = random.randint(0, len(solution_blocks) - 1)
            b_idx = solution_blocks[idx_in_sol]
            
            # Find a candidate block to swap in
            # We can pick a random block not in solution, or a block that covers some essential t-subsets.
            # For efficiency, let's pick a random block not in solution.
            # To make it smarter, we can bias towards blocks that cover essential t-subsets of b_idx.
            
            t_idxs_b = block_t_indices[b_idx]
            essential_mask = (coverage_counts[t_idxs_b] == 1)
            essential_t_idxs = t_idxs_b[essential_mask]
            
            candidates = []
            if len(essential_t_idxs) > 0:
                # Intersect blocks covering essential t-subsets
                # This is expensive if done naively every time.
                # Let's just pick a random block not in solution for simplicity in SA loop.
                # Or pick a random block that shares some points?
                # Let's stick to random non-solution block for now, SA should handle the search.
                pass
            
            # Pick random candidate
            while True:
                cand_idx = random.randint(0, n_blocks - 1)
                if not in_solution[cand_idx]:
                    break
            
            t_idxs_cand = block_t_indices[cand_idx]
            
            # Calculate delta cost for swap
            # Remove b_idx, Add cand_idx
            # Delta blocks: 0
            # Delta uncovered: 
            #   t-subsets in b_idx but not cand_idx: if count was 1, they become uncovered (+1 each)
            #   t-subsets in cand_idx but not b_idx: if count was 0, they become covered (-1 each)
            
            # Efficient calculation:
            # 1. Count t-subsets in b_idx with count 1 (become uncovered if not covered by cand)
            # 2. Count t-subsets in cand_idx with count 0 (become covered if not covered by b)
            
            # We need to know if cand covers the essential parts of b.
            # And if b covers the missing parts of cand.
            
            # Let's compute the new coverage counts hypothetically or just count changes.
            # Since k=7, t=4, each block has 35 t-subsets.
            
            # Essential in b: t in t_idxs_b where cov[t] == 1
            # If cand does NOT cover t, then t becomes uncovered.
            # Missing in cand: t in t_idxs_cand where cov[t] == 0
            # If b does NOT cover t, then t remains uncovered (no change in uncovered count from 0 to 0? 
            # Wait. If cov[t] == 0, it is uncovered. If we add cand, and cand covers t, it becomes covered.
            # So for t in t_idxs_cand with cov[t] == 0:
            #   If b covers t: no change (was uncovered, still uncovered? No, b is being removed. 
            #   Wait. Current state: b is in solution.
            #   If cov[t] == 0, it means NO block in solution covers t.
            #   So b does NOT cover t.
            #   If we remove b and add cand:
            #   If cand covers t, t becomes covered. Uncovered count decreases by 1.
            #   If cand does not cover t, t remains uncovered. No change.
            
            # So:
            # Delta_uncovered = 
            #   (Number of t in t_idxs_b with cov[t]==1 and t NOT in t_idxs_cand)  # These become uncovered
            #   - (Number of t in t_idxs_cand with cov[t]==0 and t NOT in t_idxs_b) # These become covered
            
            set_b = set(t_idxs_b.tolist())
            set_cand = set(t_idxs_cand.tolist())
            
            # t in b, cov 1, not in cand
            newly_uncovered = 0
            for t_idx in t_idxs_b:
                if coverage_counts[t_idx] == 1 and t_idx not in set_cand:
                    newly_uncovered += 1
            
            # t in cand, cov 0, not in b
            newly_covered = 0
            for t_idx in t_idxs_cand:
                if coverage_counts[t_idx] == 0 and t_idx not in set_b:
                    newly_covered += 1
            
            delta_cost = penalty * (newly_uncovered - newly_covered)
            
            if delta_cost <= 0 or (random.random() < np.exp(-delta_cost / T)):
                # Accept move
                solution_blocks[idx_in_sol] = cand_idx
                in_solution[b_idx] = False
                in_solution[cand_idx] = True
                
                coverage_counts[t_idxs_b] -= 1
                coverage_counts[t_idxs_cand] += 1
                
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_len = current_len
                    best_solution = solution_blocks.copy()
        
        T *= cooling
        
        # Occasionally restart or reset if stuck?
        # For now, simple cooling.

    final_blocks = [list(blocks[b_idx]) for b_idx in best_solution]
    return final_blocks
