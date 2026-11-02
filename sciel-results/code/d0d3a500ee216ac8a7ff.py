"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    block_cover_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_set_index[ts])
        block_cover_indices.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Precompute: for each t-set index, which blocks cover it?
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, block in enumerate(all_blocks):
        for ts_idx in block_cover_indices[b_idx]:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # Precompute block cover as numpy arrays for faster operations
    block_cover_np = [np.array(cov, dtype=np.int32) for cov in block_cover_indices]
    
    # Precompute coverage counts for pivot selection
    t_set_coverage_count = [len(t_set_to_blocks[i]) for i in range(num_t_sets)]
    
    # 3. Greedy Set Cover with multiple strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy with most-constrained pivot
    solution = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, block_cover_np,
                             t_set_to_blocks, t_set_coverage_count, num_t_sets, deadline,
                             rng_seed=42, pivot_strategy="constrained")
    if solution:
        best_size = len(solution)
        best_solution = solution
    
    # Strategy 2: Randomized greedy with different seeds
    if time.time() < deadline - 15.0:
        for seed in range(1, 8):
            if time.time() > deadline - 5.0:
                break
            sol = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, block_cover_np,
                                t_set_to_blocks, t_set_coverage_count, num_t_sets, deadline,
                                rng_seed=seed + 100, pivot_strategy="random")
            if sol and len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    # Strategy 3: Greedy with noisy gain (add small random perturbation)
    if time.time() < deadline - 15.0:
        for seed in range(1, 5):
            if time.time() > deadline - 5.0:
                break
            sol = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, block_cover_np,
                                t_set_to_blocks, t_set_coverage_count, num_t_sets, deadline,
                                rng_seed=seed + 200, pivot_strategy="constrained", noise=0.1)
            if sol and len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    # Local improvement: try to swap blocks to reduce count
    if best_solution and time.time() < deadline - 10.0:
        best_solution = _local_improve(best_solution, all_blocks, block_cover_indices, block_cover_np,
                                       t_set_to_blocks, t_set_index, v, k, t, num_t_sets, deadline)
    
    if best_solution is None:
        return []
        
    return best_solution

def _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, block_cover_np,
                  t_set_to_blocks, t_set_coverage_count, num_t_sets, deadline, 
                  rng_seed=42, pivot_strategy="constrained", noise=0.0):
    rng = random.Random(rng_seed)
    
    # Use a boolean array for uncovered (faster than set for membership)
    uncovered = np.ones(num_t_sets, dtype=bool)
    uncovered_count = num_t_sets
    selected_blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline - 2.0:
            return None  # Timeout
        
        # Choose pivot t-set
        if pivot_strategy == "constrained":
            # Pick uncovered t-set with fewest covering blocks (most constrained)
            best_pivot = -1
            best_count = float('inf')
            # Sample a few candidates for speed
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) <= 20:
                candidates = uncovered_indices
            else:
                candidates = rng.sample(uncovered_indices.tolist(), 20)
            for ts_idx in candidates:
                cnt = t_set_coverage_count[ts_idx]
                if cnt < best_count:
                    best_count = cnt
                    best_pivot = ts_idx
            if best_pivot == -1:
                best_pivot = uncovered_indices[0]
        else:  # random
            uncovered_indices = np.where(uncovered)[0]
            best_pivot = rng.choice(uncovered_indices.tolist())
        
        candidate_block_indices = t_set_to_blocks[best_pivot]
        
        best_block_idx = -1
        max_gain = 0
        
        for b_idx in candidate_block_indices:
            # Calculate gain using numpy
            cov = block_cover_np[b_idx]
            gain = np.sum(uncovered[cov])
            
            # Add noise for diversity
            if noise > 0:
                gain += rng.uniform(0, noise)
            
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
        
        # Add best block to solution
        selected_blocks.append(all_blocks[best_block_idx])
        
        # Update uncovered
        cov = block_cover_np[best_block_idx]
        newly_covered = np.sum(uncovered[cov])
        uncovered[cov] = False
        uncovered_count -= newly_covered
        
        if uncovered_count <= 0:
            break
    
    return selected_blocks

def _local_improve(solution, all_blocks, block_cover_indices, block_cover_np,
                   t_set_to_blocks, t_set_index, v, k, t, num_t_sets, deadline):
    """Try to improve the solution by removing redundant blocks or swapping."""
    if not solution:
        return solution
    
    current = list(solution)
    
    # First pass: remove blocks that are redundant (all their t-sets covered by other blocks)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Build coverage map: for each t-set, which blocks in current cover it
        t_set_covered_by = [[] for _ in range(num_t_sets)]
        for b_idx, block in enumerate(current):
            for ts_idx in block_cover_indices[b_idx]:
                t_set_covered_by[ts_idx].append(b_idx)
        
        # Try to remove each block
        new_current = []
        for b_idx, block in enumerate(current):
            if time.time() > deadline - 5.0:
                return current
            # Check if all t-sets covered by this block are also covered by other blocks
            redundant = True
            for ts_idx in block_cover_indices[b_idx]:
                others = [x for x in t_set_covered_by[ts_idx] if x != b_idx]
                if not others:
                    redundant = False
                    break
            if not redundant:
                new_current.append(block)
        
        if len(new_current) < len(current):
            current = new_current
            improved = True
    
    return current

# EVOLVE-BLOCK-END
