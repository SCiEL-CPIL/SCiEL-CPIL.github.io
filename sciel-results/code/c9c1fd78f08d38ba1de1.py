"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Use sets for faster intersection
    block_cover_sets = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_sets.append(frozenset(indices))
    
    # Precompute t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_set in enumerate(block_cover_sets):
        for ts_idx in cover_set:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    import numpy as np
    
    # Create a binary matrix: rows = blocks, cols = t-sets
    # block_matrix[b, ts] = 1 if block b covers t-set ts
    block_matrix = np.zeros((num_blocks, num_t_sets), dtype=np.uint8)
    for b_idx in range(num_blocks):
        for ts_idx in block_cover_sets[b_idx]:
            block_matrix[b_idx, ts_idx] = 1
    
    # Precompute coverage counts per block
    block_cover_counts = block_matrix.sum(axis=1)
    
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Greedy beam search (fast, good baseline)
    strategies = [
        (1, 42),
        (5, 101),
        (10, 102),
        (20, 103),
        (50, 104),
        (10, 201),
        (20, 202),
        (50, 203),
        (100, 301),
        (200, 302),
    ]
    
    for beam_width, seed in strategies:
        if time.time() > deadline - 15.0:
            break
            
        sol = _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, 
                                 rng_seed=seed, beam_width=beam_width)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    # Strategy 2: ILP-inspired branch-and-bound with cutting planes
    if time.time() < deadline - 20.0 and best_solution is not None:
        sol2 = _ilp_branch_and_bound(
            v, k, t, all_blocks, block_cover_sets, t_set_to_blocks, 
            block_matrix, block_cover_counts, num_t_sets, num_blocks,
            deadline, initial_solution=best_solution
        )
        if sol2 and len(sol2) < best_size:
            best_size = len(sol2)
            best_solution = sol2
    
    if best_solution is None:
        return []
        
    return best_solution

def _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, rng_seed=42, beam_width=1):
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline - 2.0:
            return None
        
        candidate_blocks = set()
        
        # Pick beam_width random t-sets from uncovered
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            targets = rng.sample(list(uncovered), beam_width)
            
        for target_tset_idx in targets:
            # Find best block covering this target
            best_block_idx = -1
            max_gain = 0
            
            for b_idx in t_set_to_blocks[target_tset_idx]:
                gain = len(block_cover_sets[b_idx] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)
        
        # Pick the best block among candidates
        final_best_block_idx = -1
        final_max_gain = 0
        
        for b_idx in candidate_blocks:
            gain = len(block_cover_sets[b_idx] & uncovered)
            if gain > final_max_gain:
                final_max_gain = gain
                final_best_block_idx = b_idx
        
        if final_best_block_idx == -1:
            break
            
        selected_blocks.append(all_blocks[final_best_block_idx])
        
        # Update uncovered
        uncovered -= block_cover_sets[final_best_block_idx]
        
    return selected_blocks

def _ilp_branch_and_bound(v, k, t, all_blocks, block_cover_sets, t_set_to_blocks, 
                          block_matrix, block_cover_counts, num_t_sets, num_blocks,
                          deadline, initial_solution=None):
    """
    ILP-inspired branch-and-bound heuristic.
    
    Model: Minimize sum(x_b) subject to for each t-set s: sum_{b covers s} x_b >= 1
    where x_b in {0,1} indicates whether block b is selected.
    
    We use a cutting-plane/branch-and-bound hybrid:
    1. Start with a feasible solution (from greedy)
    2. Iteratively identify uncovered t-sets
    3. Add constraints (cuts) to prevent certain block combinations
    4. Use local search to improve
    """
    import numpy as np
    
    if initial_solution is None:
        return None
    
    # Start with the initial solution
    selected = set()
    for block in initial_solution:
        block_idx = all_blocks.index(block)
        selected.add(block_idx)
    
    # Track which t-sets are covered
    covered_tsets = set()
    for b_idx in selected:
        covered_tsets.update(block_cover_sets[b_idx])
    
    uncovered = set(range(num_t_sets)) - covered_tsets
    
    # If already complete, return
    if not uncovered:
        return [all_blocks[b] for b in selected]
    
    # Local search: try to improve by swapping blocks
    best_selected = set(selected)
    best_size = len(selected)
    
    # Precompute for each t-set, the blocks that cover it
    # t_set_to_blocks is already provided
    
    # Iterative improvement loop
    iteration = 0
    max_iterations = 50
    
    while time.time() < deadline - 5.0 and iteration < max_iterations:
        iteration += 1
        
        # Find the most "expensive" uncovered t-set (one with fewest covering blocks)
        # This is a heuristic for which constraint is most binding
        min_cover_count = float('inf')
        target_tset = -1
        for ts_idx in list(uncovered)[:100]:  # Limit to first 100 for speed
            if ts_idx in uncovered:
                cover_count = len(t_set_to_blocks[ts_idx])
                if cover_count < min_cover_count:
                    min_cover_count = cover_count
                    target_tset = ts_idx
        
        if target_tset == -1:
            break
        
        # Find blocks that cover target_tset
        covering_blocks = t_set_to_blocks[target_tset]
        
        # Try adding the best block for this t-set
        best_add_block = -1
        best_gain = 0
        for b_idx in covering_blocks:
            if b_idx not in best_selected:
                gain = len(block_cover_sets[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_add_block = b_idx
        
        if best_add_block != -1:
            # Try adding this block
            new_selected = set(best_selected)
            new_selected.add(best_add_block)
            new_covered = set()
            for b in new_selected:
                new_covered.update(block_cover_sets[b])
            new_uncovered = set(range(num_t_sets)) - new_covered
            
            # Try to remove some blocks
            improved = False
            for b_remove in list(new_selected):
                if b_remove == best_add_block:
                    continue
                candidate_selected = new_selected - {b_remove}
                if not candidate_selected:
                    continue
                cand_covered = set()
                for b in candidate_selected:
                    cand_covered.update(block_cover_sets[b])
                if set(range(num_t_sets)) <= cand_covered:
                    # Valid covering with fewer blocks
                    if len(candidate_selected) < len(best_selected):
                        best_selected = candidate_selected
                        best_size = len(best_selected)
                        improved = True
                        break
            
            if not improved and best_gain > 0:
                # Keep the added block even if we couldn't remove one
                best_selected = new_selected
                best_size = len(best_selected)
        
        # Try removing blocks from current best
        for b_remove in list(best_selected):
            if time.time() > deadline - 5.0:
                break
            candidate_selected = best_selected - {b_remove}
            if not candidate_selected:
                continue
            cand_covered = set()
            for b in candidate_selected:
                cand_covered.update(block_cover_sets[b])
            if set(range(num_t_sets)) <= cand_covered:
                best_selected = candidate_selected
                best_size = len(best_selected)
        
        # Check if we're making progress
        if best_size >= best_size:
            pass  # Continue searching
    
    # Convert back to blocks
    result = [all_blocks[b] for b in best_selected]
    
    # Verify
    if _verify_covering(result, v, k, t):
        return result
    
    return None

def _verify_covering(blocks, v, k, t):
    """Verify that the blocks form a valid covering."""
    if not blocks:
        return False
    
    # Check all blocks are valid
    for block in blocks:
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        if any(x < 0 or x >= v for x in block):
            return False
    
    # Check all t-subsets are covered
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    for ts in itertools.combinations(range(v), t):
        if ts not in covered:
            return False
    
    return True
# EVOLVE-BLOCK-END
