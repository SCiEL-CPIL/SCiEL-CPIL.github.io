import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # Using uint8 for memory efficiency
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    rng = random.Random(42)

    # Helper: Greedy construction
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            # Gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            # Update uncovered
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    # Helper: Remove redundant blocks iteratively
    def remove_redundant(indices):
        indices_set = set(indices)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in indices_set:
                row = M[idx]
                covered = np.where(row == 1)[0]
                # If all covered tsubs have count > 1, it's redundant
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set), counts

    # Helper: Local search with swaps
    def local_search(indices):
        indices_set = set(indices)
        # Recalculate counts from scratch to be safe, or pass in
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        is_selected = np.zeros(num_kblocks, dtype=bool)
        for idx in indices_set:
            is_selected[idx] = True
            
        improved = True
        max_iter = 50 # Limit iterations per local search call
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 10.0:
                break
            
            # Pass 1: Remove redundant
            new_indices, counts = remove_redundant(list(indices_set))
            if len(new_indices) < len(indices_set):
                # Update sets and counts
                indices_set = set(new_indices)
                is_selected = np.zeros(num_kblocks, dtype=bool)
                for idx in indices_set:
                    is_selected[idx] = True
                improved = True
                continue
            
            # Pass 2: Try swaps
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.time() > deadline - 10.0:
                    return list(indices_set)
                
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                critical_mask = counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # Redundant, should have been removed, but just in case
                    indices_set.remove(b_idx)
                    counts -= row
                    is_selected[b_idx] = False
                    improved = True
                    break
                
                # We need a block that covers all critical_indices
                # Skip if too many constraints to make it fast
                if len(critical_indices) > 10:
                    continue
                    
                # Find candidates
                # M[:, critical_indices].all(axis=1)
                candidates_mask = M[:, critical_indices].all(axis=1)
                candidates_mask &= ~is_selected
                
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    c_idx = candidates[rng.randrange(len(candidates))]
                    
                    # Perform swap
                    indices_set.remove(b_idx)
                    indices_set.add(c_idx)
                    
                    counts -= M[b_idx]
                    counts += M[c_idx]
                    
                    is_selected[b_idx] = False
                    is_selected[c_idx] = True
                    
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 1: Initial Greedy
    initial_sol = greedy_construction()
    if not initial_sol:
        return []
        
    best_solution = initial_sol
    best_len = len(best_solution)
    
    # Phase 2: Local Search / ILS
    current_sol = best_solution[:]
    
    # Run local search multiple times with perturbations if time permits
    # Since the problem is static, we can just run a deep local search
    # or multiple independent local searches from perturbed starts.
    
    # First, polish the greedy solution
    polished_sol = local_search(current_sol)
    if len(polished_sol) < best_len:
        best_len = len(polished_sol)
        best_solution = polished_sol[:]
        
    # Try to improve further with random restarts/perturbations
    # Given the time limit, we'll do a few iterations
    for i in range(10):
        if time.time() > deadline - 15.0:
            break
            
        # Perturb current best: remove a few random blocks and repair
        # Or just shuffle and run local search again?
        # A simple perturbation: remove 5 random blocks, then repair greedily
        if len(best_solution) > 20:
            sol_copy = best_solution[:]
            # Remove up to 5 blocks
            num_remove = min(5, len(sol_copy))
            for _ in range(num_remove):
                if sol_copy:
                    rm_idx = rng.randrange(len(sol_copy))
                    sol_copy.pop(rm_idx)
            
            # Repair: Greedily add blocks to cover uncovered
            # We need to know what's covered by sol_copy
            counts = np.zeros(num_tsubs, dtype=np.int32)
            for idx in sol_copy:
                counts += M[idx]
            
            uncovered_mask = counts == 0
            if not np.any(uncovered_mask):
                # Still a valid covering, just smaller? No, we removed blocks.
                # If no uncovered, it's a valid smaller covering.
                if len(sol_copy) < best_len:
                    best_len = len(sol_copy)
                    best_solution = sol_copy[:]
                continue
            
            # Greedy repair
            uncovered = uncovered_mask.astype(np.uint8)
            selected_indices = set(sol_copy)
            
            while np.any(uncovered):
                if time.time() > deadline - 10.0:
                    break
                gains = M @ uncovered
                # Exclude already selected
                gains[list(selected_indices)] = 0
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                selected_indices.add(best_idx)
                row = M[best_idx]
                covered_indices = np.where(row == 1)[0]
                uncovered[covered_indices] = 0
                
            candidate_sol = list(selected_indices)
            # Polish
            candidate_sol = local_search(candidate_sol)
            
            if len(candidate_sol) < best_len:
                best_len = len(candidate_sol)
                best_solution = candidate_sol[:]
        
    final_blocks = get_blocks(best_solution)
    return final_blocks
