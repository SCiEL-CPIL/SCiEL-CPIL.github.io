import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    # Precompute t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Precompute coverage matrix: for each block, indices of t-subsets it covers
    t_per_block = comb(k, t)
    coverage_matrix = np.zeros((n_k, t_per_block), dtype=np.int32)
    
    for i, block in enumerate(k_subs):
        subs_in_block = itertools.combinations(block, t)
        indices = [t_sub_idx[sub] for sub in subs_in_block]
        coverage_matrix[i] = indices
        
    # State: which t-subsets are not yet covered
    is_uncovered = np.ones(n_t, dtype=bool)
    
    blocks_found = []
    
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s for post-processing
    
    while np.any(is_uncovered):
        if time.time() > deadline:
            break
            
        taken = np.take(is_uncovered, coverage_matrix)
        gains = np.sum(taken, axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Random tie-breaking to allow diversity if run multiple times?
        # But we only run once. Deterministic is fine.
        best_idx = np.argmax(gains)
        best_block = k_subs[best_idx]
        
        blocks_found.append(list(best_block))
        
        is_uncovered[coverage_matrix[best_idx]] = False
        
    # Post-processing: Remove redundant blocks
    # Build inverse index: t_sub_idx -> list of block indices in blocks_found
    # We only care about blocks in blocks_found
    block_indices_in_solution = list(range(len(blocks_found)))
    
    # Map solution block index to original k_sub index
    # Actually, we just need to check coverage among the selected blocks.
    
    # For each t-subset, which selected blocks cover it?
    # selected_block_to_t_subs is just the rows of coverage_matrix corresponding to blocks_found
    # But we need the reverse.
    
    # Let's build a set of t-subsets covered by each selected block index
    # And a dict: t_sub_idx -> set of selected block indices covering it
    
    t_sub_coverers = {j: set() for j in range(n_t)}
    
    for sol_idx, block in enumerate(blocks_found):
        # Get the indices of t-subsets covered by this block
        # We need the original k_sub index to look up in coverage_matrix
        # Wait, blocks_found contains the actual points.
        # We need to map back to the original k_subs index or recompute.
        # It's better to store the original index during greedy.
        
        pass # Redo greedy to store indices

    # Let's restart the greedy part to store indices
    # Or just re-map.
    # k_subs is a list of tuples. We can create a dict tuple->index?
    # No, just store the index in blocks_found_indices.

    # Re-implementing the loop to capture indices is safer.
    
    is_uncovered = np.ones(n_t, dtype=bool)
    blocks_found_indices = []
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    while np.any(is_uncovered):
        if time.time() > deadline:
            break
            
        taken = np.take(is_uncovered, coverage_matrix)
        gains = np.sum(taken, axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        best_idx = np.argmax(gains)
        
        blocks_found_indices.append(best_idx)
        
        is_uncovered[coverage_matrix[best_idx]] = False
        
    # Now blocks_found_indices contains indices into k_subs and coverage_matrix
    
    # Build reverse map for redundancy check
    # t_sub_idx -> set of solution indices (position in blocks_found_indices)
    t_sub_coverers = [set() for _ in range(n_t)]
    
    for sol_pos, orig_idx in enumerate(blocks_found_indices):
        t_indices = coverage_matrix[orig_idx]
        for t_idx in t_indices:
            t_sub_coverers[t_idx].add(sol_pos)
            
    # Check for redundancy
    # A block at sol_pos is redundant if for every t_idx it covers, 
    # there is another sol_pos' in t_sub_coverers[t_idx] such that sol_pos' != sol_pos
    
    redundant = [False] * len(blocks_found_indices)
    
    for sol_pos in range(len(blocks_found_indices)):
        orig_idx = blocks_found_indices[sol_pos]
        t_indices = coverage_matrix[orig_idx]
        is_redundant = True
        for t_idx in t_indices:
            # Check if any other block covers this t_idx
            # t_sub_coverers[t_idx] contains sol_pos and possibly others
            if len(t_sub_coverers[t_idx]) == 1:
                is_redundant = False
                break
        if is_redundant:
            redundant[sol_pos] = True
            # Remove sol_pos from all coverers
            for t_idx in t_indices:
                t_sub_coverers[t_idx].remove(sol_pos)
                
    # Filter blocks
    final_blocks = [k_subs[idx] for idx, is_red in zip(blocks_found_indices, redundant) if not is_red]
    
    return [list(b) for b in final_blocks]
