import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill
    
    D = 11
    candidates = []
    
    # Generate all integer vectors with norm squared <= 4
    # Type 1: Weight 1, value +/- 1 (norm sq 1)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Weight 1, value +/- 2 (norm sq 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: Weight 2, values +/- 1 (norm sq 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 4: Weight 3, values +/- 1 (norm sq 3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 5: Weight 4, values +/- 1 (norm sq 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4
    
    # Compute conflict matrix: conflict if dist_sq < max_norm_sq
    # dist_sq = ||u||^2 + ||v||^2 - 2*u.v
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Strategy: Use a combination of greedy MIS with local search (simulated annealing style)
    # The key insight from inspirations is that the problem is MIS on the conflict graph.
    # We'll use a more sophisticated approach:
    # 1. Start with greedy MIS
    # 2. Improve with local search: try to add back removed vertices or swap
    # 3. Use spectral information to guide the search (quantum-inspired idea)
    
    # Precompute degree
    degrees = conflict.sum(axis=1).astype(np.int64)
    
    def greedy_mis(conflict, degrees, n, seed=0, time_limit=10.0):
        """Greedy MIS with randomization among min-degree vertices."""
        rng = np.random.RandomState(seed)
        remaining = np.ones(n, dtype=bool)
        deg = degrees.copy()
        selected = []
        
        while remaining.any():
            if time.time() > time_limit:
                break
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = deg[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = rng.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            to_remove = np.zeros(n, dtype=bool)
            to_remove[chosen_global] = True
            to_remove[conflict[chosen_global]] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees
                for idx in rem_indices:
                    if remaining[idx]:
                        deg[idx] -= np.sum(conflict[actual_remove, idx])
            
            remaining[actual_remove] = False
        
        return selected
    
    def local_search_improve(selected, conflict, n, max_iters=5000, time_limit=10.0):
        """Try to improve the independent set by adding vertices or swapping."""
        selected_set = set(selected)
        selected_arr = np.array(selected)
        
        # Build adjacency for fast lookup
        # We want to try adding a vertex not in the set if it has no conflicts with the set
        # Or swap: remove one from set, add two that don't conflict with new set
        
        in_set = np.zeros(n, dtype=bool)
        in_set[selected] = True
        
        # For each vertex not in set, check if it can be added
        # A vertex v can be added if no vertex in selected_set conflicts with v
        # i.e., for all u in selected, not conflict[v, u]
        
        # Precompute: for each vertex v, the number of conflicts with the current set
        # conflicts_with_set[v] = sum(conflict[v, in_set])
        conflicts_with_set = (conflict @ in_set).astype(np.int64)
        
        improved = True
        iters = 0
        start_ls = time.time()
        
        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1
            
            # Try adding a vertex with 0 conflicts with the set
            candidates_add = np.where((conflicts_with_set == 0) & (~in_set))[0]
            if len(candidates_add) > 0:
                # Pick one at random
                v = candidates_add[np.random.randint(len(candidates_add))]
                selected.append(v)
                in_set[v] = True
                # Update conflicts_with_set for all vertices
                conflicts_with_set += conflict[v]
                improved = True
                continue
            
            # Try swapping: remove u from set, add v1 and v2 that don't conflict with new set
            # This is more expensive, so do it less frequently
            if iters % 10 == 0 and len(selected) > 10:
                # Pick a random vertex u to remove
                u = selected[np.random.randint(len(selected))]
                
                # Temporarily remove u
                in_set[u] = False
                conflicts_with_set -= conflict[u]
                selected.remove(u)
                
                # Now try to add two vertices
                added = []
                for _ in range(2):
                    candidates_add = np.where((conflicts_with_set == 0) & (~in_set))[0]
                    if len(candidates_add) == 0:
                        break
                    v = candidates_add[np.random.randint(len(candidates_add))]
                    added.append(v)
                    in_set[v] = True
                    conflicts_with_set += conflict[v]
                
                if len(added) == 2:
                    selected.extend(added)
                    improved = True
                else:
                    # Revert: remove added, re-add u
                    for v in added:
                        in_set[v] = False
                        conflicts_with_set -= conflict[v]
                        if v in selected:
                            selected.remove(v)
                    in_set[u] = True
                    conflicts_with_set += conflict[u]
                    selected.append(u)
        
        return selected
    
    # Run multiple restarts with time guard
    attempt = 0
    while time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 1:
            break
        
        # Allocate time between greedy and local search
        greedy_time = min(3.0, remaining_time * 0.3)
        ls_time = min(7.0, remaining_time * 0.6)
        
        # Greedy MIS
        selected = greedy_mis(conflict, degrees, n, seed=attempt, time_limit=greedy_time)
        
        # Local search improvement
        selected = local_search_improve(selected, conflict, n, max_iters=10000, time_limit=ls_time)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
        
        attempt += 1
    
    # If no valid selection was found (shouldn't happen), return a single vector
    if len(best_selected) == 0:
        best_selected = [0]
    
    result = candidates[best_selected]
    return result
