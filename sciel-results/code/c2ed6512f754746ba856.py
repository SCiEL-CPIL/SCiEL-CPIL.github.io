import itertools
import time
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block for faster access later
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Phase 1: Greedy Set Cover
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 60.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = np.argmax(gains)
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Phase 2: Local Search Deletion
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    current_best = len(selected_block_indices)
    
    # Phase 3: Try ILP refinement on the current solution
    if time.time() < deadline - 20.0:
        try:
            # Objective: minimize sum(x)
            c = np.ones(num_blocks)
            
            # Constraints: A^T x >= 1
            A_T = A.T.tocsr()
            
            constraint = LinearConstraint(A_T, lb=1, ub=np.inf)
            
            # Bounds: x in [0, 1]
            lb = np.zeros(num_blocks)
            ub = np.ones(num_blocks)
            bounds = Bounds(lb, ub)
            
            # Integrality: all variables are binary
            integrality = np.ones(num_blocks)
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                       options={'time_limit': 10.0})
            
            if res.status == 0 and res.x is not None:
                ilp_solution = res.x.astype(int)
                ilp_blocks = np.where(ilp_solution == 1)[0]
                
                # Verify validity
                if len(ilp_blocks) > 0:
                    cover_check = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in ilp_blocks:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_check[covered] += 1
                    
                    if np.all(cover_check >= 1) and len(ilp_blocks) < current_best:
                        selected_block_indices = list(ilp_blocks)
                        current_best = len(selected_block_indices)
                        
        except Exception:
            pass
            
    # Phase 4: Local Search with 1-for-1 swaps and perturbation
    if time.time() < deadline - 5.0:
        # Build a set for fast lookup of selected blocks
        selected_set = set(selected_block_indices)
        
        # Precompute for each t-subset which blocks cover it (for quick lookup)
        # We'll use a list of lists: t_subset_coverers[t_idx] = list of block indices covering it
        # This is expensive to build but useful for swap operations
        # Actually, let's just use the A matrix indirectly
        
        # For swap: remove block i, add block j
        # We need to check that all t-subsets covered by i are still covered by others + j
        
        # Let's do a simpler approach: try random swaps
        # Pick a random selected block, try to replace it with a random unselected block
        # Check if the cover is still valid
        
        # To make this efficient, we'll track cover counts
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] += 1
        
        # Try multiple restarts of local search
        best_solution = list(selected_block_indices)
        best_size = len(best_solution)
        
        # Number of iterations based on remaining time
        max_iters = 5000
        iters = 0
        
        while time.time() < deadline - 5.0 and iters < max_iters:
            iters += 1
            
            if len(selected_block_indices) >= best_size:
                # Reset to best if we've grown
                if len(selected_block_indices) > best_size:
                    selected_block_indices = list(best_solution)
                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in selected_block_indices:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] += 1
                else:
                    # Same size, continue searching
                    pass
            
            if not selected_block_indices:
                break
                
            # Pick a random block to potentially remove
            i = random.randint(0, len(selected_block_indices) - 1)
            b_remove = selected_block_indices[i]
            
            # Get t-subsets that would become uncovered if we remove b_remove
            covered_remove = A.indices[A.indptr[b_remove]:A.indptr[b_remove+1]]
            # These are t-subsets with count == 1 after removal would be 0
            # Actually, we need to find t-subsets that are ONLY covered by b_remove
            # i.e., cover_counts[c] == 1 for c in covered_remove
            
            # Find critical t-subsets (only covered by b_remove)
            critical = []
            for c in covered_remove:
                if cover_counts[c] == 1:
                    critical.append(c)
            
            if not critical:
                # b_remove is redundant, just remove it
                selected_block_indices.pop(i)
                cover_counts[covered_remove] -= 1
                if len(selected_block_indices) < best_size:
                    best_solution = list(selected_block_indices)
                    best_size = len(best_solution)
                continue
            
            # We need to find a block that covers all critical t-subsets
            # This is expensive, so let's sample a few random blocks
            
            # Sample up to 5 random blocks to try
            tried = set()
            found_swap = False
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                b_add = random.randint(0, num_blocks - 1)
                if b_add in selected_set:
                    continue
                if b_add in tried:
                    continue
                tried.add(b_add)
                
                # Check if b_add covers all critical t-subsets
                covered_add = set(A.indices[A.indptr[b_add]:A.indptr[b_add+1]])
                if all(c in covered_add for c in critical):
                    # Perform the swap
                    selected_block_indices[i] = b_add
                    selected_set.remove(b_remove)
                    selected_set.add(b_add)
                    cover_counts[covered_remove] -= 1
                    for c in A.indices[A.indptr[b_add]:A.indptr[b_add+1]]:
                        cover_counts[c] += 1
                    found_swap = True
                    break
            
            if not found_swap:
                # Perturbation: randomly add a block
                if len(selected_block_indices) < best_size + 2:
                    b_add = random.randint(0, num_blocks - 1)
                    if b_add not in selected_set:
                        selected_block_indices.append(b_add)
                        selected_set.add(b_add)
                        for c in A.indices[A.indptr[b_add]:A.indptr[b_add+1]]:
                            cover_counts[c] += 1
                        if len(selected_block_indices) < best_size:
                            best_solution = list(selected_block_indices)
                            best_size = len(selected_block_indices)
        
        # Restore best solution
        if len(selected_block_indices) > best_size:
            selected_block_indices = best_solution
        
    # Phase 5: Final Deletion Pass on the best solution found
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
