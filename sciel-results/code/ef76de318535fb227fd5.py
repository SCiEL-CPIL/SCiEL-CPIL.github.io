"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.float64)
    n_candidates = len(candidates)
    
    # Precompute Gram matrix: G[i,j] = candidates[i] · candidates[j]
    # Norm squared = 4 for all
    # Constraint: |u-v|^2 >= 4  <=>  8 - 2*u·v >= 4  <=>  u·v <= 2
    G = np.dot(candidates, candidates.T)
    
    # Build adjacency: A[i,j] = 1 if candidates[i] and candidates[j] are incompatible (dot > 2)
    A = (G > 2 + 1e-9).astype(np.int8)
    np.fill_diagonal(A, 0)
    
    # Precompute neighbor lists for each node
    neighbor_lists = [np.where(A[i] == 1)[0] for i in range(n_candidates)]
    
    # Degree of each node
    degrees = np.array([len(nl) for nl in neighbor_lists])
    
    def grasp_construct(rng, time_limit, alpha=0.5):
        """
        GRASP construction phase: build an independent set using a restricted candidate list.
        At each step, from the set of feasible candidates (not conflicting with selected),
        pick one from the top alpha fraction of the best candidates (by some heuristic).
        
        Heuristic: prefer candidates with lower degree (fewer conflicts) among feasible ones.
        """
        selected = set()
        in_selected = np.zeros(n_candidates, dtype=bool)
        
        while True:
            if time.time() > time_limit:
                break
            
            # Find all feasible candidates (not in selected, not conflicting with selected)
            if selected:
                # A candidate is feasible if it's not in selected and doesn't conflict with any selected
                conflict_mask = np.zeros(n_candidates, dtype=bool)
                for s in selected:
                    conflict_mask[neighbor_lists[s]] = True
                conflict_mask[list(selected)] = True
                feasible = np.where(~conflict_mask)[0]
            else:
                feasible = np.arange(n_candidates)
            
            if len(feasible) == 0:
                break
            
            # Score feasible candidates: lower degree is better
            # Use negative degree as score (higher score = better)
            scores = -degrees[feasible].astype(np.float64)
            
            # Add small random perturbation to break ties
            scores += rng.random(len(feasible)) * 0.1
            
            # Get the best score
            max_score = scores.max()
            min_score = scores.min()
            
            # Restricted candidate list: candidates with score >= max_score - alpha * (max_score - min_score)
            if max_score - min_score < 1e-10:
                rcl = feasible
            else:
                threshold = max_score - alpha * (max_score - min_score)
                rcl = feasible[scores >= threshold]
            
            if len(rcl) == 0:
                rcl = feasible
            
            # Pick randomly from RCL
            pick_idx = rng.randint(len(rcl))
            chosen = rcl[pick_idx]
            
            selected.add(chosen)
            in_selected[chosen] = True
        
        return list(selected)
    
    def swap_local_search(sel_list, rng, time_limit, max_iters=5000):
        """
        1-for-1 swap local search: try to improve the independent set by
        removing one element and adding one (or more) that become feasible.
        
        Strategy: for each element in the set, try removing it and then greedily
        adding back elements that were previously blocked.
        """
        selected = set(sel_list)
        in_selected = np.zeros(n_candidates, dtype=bool)
        for s in selected:
            in_selected[s] = True
        
        best_size = len(selected)
        improved = True
        iters = 0
        
        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            iters += 1
            
            # For each element in the current set, try removing it
            for s in list(selected):
                if time.time() > time_limit:
                    break
                
                # Remove s
                selected.remove(s)
                in_selected[s] = False
                new_size = len(selected)
                
                # Try to add elements that were previously blocked by s
                # These are candidates that conflict with s but not with remaining selected
                blocked_by_s = neighbor_lists[s]
                
                # Find candidates that conflict with s, are not in selected,
                # and don't conflict with any remaining selected
                candidates_to_try = []
                for c in blocked_by_s:
                    if in_selected[c]:
                        continue
                    # Check if c conflicts with any remaining selected
                    conflicts_with_remaining = False
                    for ns in neighbor_lists[c]:
                        if in_selected[ns]:
                            conflicts_with_remaining = True
                            break
                    if not conflicts_with_remaining:
                        candidates_to_try.append(c)
                
                # Sort by degree (prefer lower degree)
                candidates_to_try.sort(key=lambda x: degrees[x])
                
                # Greedily add as many as possible
                added = 0
                for c in candidates_to_try:
                    if time.time() > time_limit:
                        break
                    # Re-check feasibility (in case earlier additions blocked this)
                    feasible = True
                    for ns in neighbor_lists[c]:
                        if in_selected[ns]:
                            feasible = False
                            break
                    if feasible:
                        selected.add(c)
                        in_selected[c] = True
                        added += 1
                
                new_size = len(selected)
                
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                    # Don't restore s; keep the new solution
                else:
                    # Restore s
                    selected.add(s)
                    in_selected[s] = True
        
        return list(selected)
    
    def grasp_iterate(rng, time_limit, alpha=0.5):
        """One GRASP iteration: construct + local search."""
        # Construction
        sol = grasp_construct(rng, time_limit, alpha)
        
        # Local search
        sol = swap_local_search(sol, rng, time_limit)
        
        return sol
    
    # GRASP main loop
    best_selected = []
    rng = np.random.RandomState(42)
    
    # Different alpha values for diversity
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # First, do a simple greedy to get a baseline
    order = list(range(n_candidates))
    order.sort(key=lambda i: degrees[i])
    baseline = []
    in_baseline = set()
    for idx in order:
        compatible = True
        for s in in_baseline:
            if A[idx, s]:
                compatible = False
                break
        if compatible:
            baseline.append(idx)
            in_baseline.add(idx)
    best_selected = baseline
    
    # GRASP iterations
    n_iterations = 0
    while time.time() < deadline - 5.0:
        n_iterations += 1
        alpha = alphas[n_iterations % len(alphas)]
        
        # Give each iteration a time budget
        iter_time = deadline - 5.0 - time.time()
        if iter_time < 1.0:
            break
        
        # Split time between construction and local search
        construct_time = time.time() + min(2.0, iter_time * 0.3)
        search_time = deadline - 5.0
        
        sol = grasp_iterate(rng, search_time, alpha)
        
        if len(sol) > len(best_selected):
            best_selected = sol
            # Print progress (optional, removed for production)
            # print(f"GRASP iter {n_iterations}: size {len(sol)}")
    
    if not best_selected:
        # Fallback: return a minimal valid set
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    return candidates[best_selected].astype(np.int64)
# EVOLVE-BLOCK-END
