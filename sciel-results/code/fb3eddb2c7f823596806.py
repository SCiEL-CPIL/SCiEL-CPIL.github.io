import numpy as np
import itertools
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute conflict matrix: conflict if dist_sq < 4
    # dist_sq = norms_sq[i] + norms_sq[j] - 2*dot(i,j)
    # We need dist_sq >= 4 for all pairs in the set (since max norm is 2 when T2/T3 present)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Precompute neighbor lists for faster updates
    # Use a bitset representation for the conflict graph
    # n=5522, so we can use a list of arrays of neighbor indices
    neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    best_selected = []
    best_size = 0
    np.random.seed(42)
    
    # Strategy: Multiple greedy passes with different tie-breaking, then local search
    n_attempts = 0
    while time.time() < deadline - 30:  # leave 30s for local search
        n_attempts += 1
        
        # Randomized greedy: pick node with min degree among remaining
        remaining = np.ones(n, dtype=bool)
        rem_count = n
        degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)
        selected = []
        
        while rem_count > 0:
            if time.time() > deadline - 30:
                break
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            rem_count -= 1
            
            # Remove chosen and its neighbors
            neigh = neighbors[chosen_global]
            to_remove = np.concatenate(([chosen_global], neigh))
            to_remove_unique = np.unique(to_remove)
            to_remove_mask = np.isin(rem_indices, to_remove_unique)
            actual_remove = to_remove_mask
            
            # Update degrees
            for idx in to_remove_unique:
                if remaining[idx]:
                    for nb in neighbors[idx]:
                        if remaining[nb]:
                            degrees[nb] -= 1
            
            remaining[to_remove_unique] = False
            rem_count -= np.sum(actual_remove) - 1
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
        
        # Local search: try to improve by swapping
        if time.time() > deadline - 10:
            break
        
        # Convert to set for fast lookup
        sel_set = set(selected)
        
        # Try random swaps: remove one selected, add one non-selected
        improved = True
        max_swaps = 500
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline - 10:
            improved = False
            # Sample some selected nodes to remove
            for _ in range(min(20, len(selected))):
                if time.time() > deadline - 10:
                    break
                idx_remove = np.random.choice(selected)
                # Find candidates to add: non-selected nodes not conflicting with (selected - {idx_remove})
                sel_minus = [s for s in selected if s != idx_remove]
                # Check which non-selected nodes can be added
                # A node can be added if it doesn't conflict with any in sel_minus
                # And it's not already selected
                non_sel = np.setdiff1d(np.arange(n), np.array(selected))
                if len(non_sel) == 0:
                    break
                
                # For each candidate, check if it conflicts with sel_minus
                # We want to find a candidate that doesn't conflict with sel_minus
                # and has high "benefit" (low degree in remaining graph)
                
                # Quick check: for each non_sel node, count conflicts with sel_minus
                # This is expensive, so sample
                sample_size = min(100, len(non_sel))
                sampled = np.random.choice(non_sel, size=sample_size, replace=False)
                
                best_candidate = -1
                best_score = -1
                
                for cand in sampled:
                    # Check if cand conflicts with any in sel_minus
                    cand_conflicts = neighbors[cand]
                    conflict_count = np.sum(np.isin(cand_conflicts, np.array(sel_minus)))
                    if conflict_count == 0:
                        # This candidate can replace idx_remove
                        # Score: how many new connections does it have? (we want low degree)
                        score = len(neighbors[cand])
                        if score < best_score or best_candidate == -1:
                            best_candidate = cand
                            best_score = score
                
                if best_candidate != -1:
                    selected.remove(idx_remove)
                    selected.append(best_candidate)
                    improved = True
                    swaps_done += 1
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    if not best_selected:
        # Fallback: all Type 1 vectors (first 220)
        best_selected = list(range(220))
    
    result = candidates[best_selected]
    return result
