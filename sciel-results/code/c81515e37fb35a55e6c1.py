"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit

    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)

    # Constraint: for vectors of norm 2 (norm_sq=4),
    # |u-v|^2 >= 4  <=>  4+4-2*u.v >= 4  <=>  u.v <= 2
    # So we need dot product <= 2 for all pairs

    # Precompute dot product matrix for all candidates
    dots = candidates @ candidates.T

    # Build incompatibility graph: edge if dot > 2
    # We want to find a large independent set (no edges)
    # incompatibility[i,j] = True if dot[i,j] > 2
    incompat = (dots > 2).astype(np.int8)
    np.fill_diagonal(incompat, 0)

    # Use spectral method: find Fiedler vector of the incompatibility graph
    # Nodes with high absolute value in Fiedler vector tend to be in large independent sets
    # Actually, for maximum independent set, we can use the eigenvector of the adjacency matrix
    # corresponding to the largest eigenvalue - nodes with high positive values are likely in the clique
    
    # Alternative: use greedy with multiple strategies
    # Strategy 1: Greedy by degree (pick node with fewest incompatibilities first)
    # Strategy 2: Random restarts with different orderings
    # Strategy 3: Local search (swap operations)

    best_set = set()
    best_size = 0

    # Compute degrees
    degrees = incompat.sum(axis=1)

    # Strategy 1: Greedy by minimum degree
    order = np.argsort(degrees)
    selected = []
    selected_set = set()
    for i in order:
        if i in selected_set:
            continue
        if not selected:
            selected.append(i)
            selected_set.add(i)
            continue
        sel_arr = np.array(selected)
        if np.all(incompat[i, sel_arr] == 0):
            selected.append(i)
            selected_set.add(i)

    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)

    # Strategy 2: Multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(500):
        if time.time() > deadline - 15.0:
            break
        
        # Random ordering with slight bias towards low-degree nodes
        weights = 1.0 / (degrees + 1)
        weights = weights / weights.sum()
        order = rng.choice(n_candidates, size=n_candidates, replace=False, p=weights)
        
        selected = []
        selected_set = set()
        for i in order:
            if i in selected_set:
                continue
            if not selected:
                selected.append(i)
                selected_set.add(i)
                continue
            sel_arr = np.array(selected)
            if np.all(incompat[i, sel_arr] == 0):
                selected.append(i)
                selected_set.add(i)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            # Early termination if we hit a good target
            if best_size >= 800:
                break

    # Strategy 3: Local search - try to add nodes that are close to being compatible
    # Or try swap operations to improve the set
    if time.time() < deadline - 10.0:
        current_set = list(best_set)
        current_set_set = set(current_set)
        
        # Try adding nodes that have only 1 or 2 conflicts
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Find nodes not in set that have minimal conflicts with set
            if current_set:
                cur_arr = np.array(current_set)
                conflicts = incompat[:, cur_arr].sum(axis=1)
            else:
                conflicts = np.zeros(n_candidates, dtype=np.int64)
            
            # Try nodes with 0 conflicts first (shouldn't happen if set is maximal)
            # Then try nodes with 1 conflict - if we remove one conflicting node and add this one
            for n_conflicts in [0, 1, 2]:
                candidates_to_add = np.where((conflicts == n_conflicts) & 
                                              (~np.isin(np.arange(n_candidates), list(current_set_set))))[0]
                
                for candidate_idx in candidates_to_add:
                    if time.time() > deadline - 5.0:
                        break
                    
                    if n_conflicts == 0:
                        current_set.append(candidate_idx)
                        current_set_set.add(candidate_idx)
                        improved = True
                        break
                    
                    # For n_conflicts > 0, try swapping: remove one conflicting node, add candidate
                    if current_set:
                        cur_arr = np.array(current_set)
                        conflict_nodes = cur_arr[incompat[candidate_idx, cur_arr] > 0]
                        
                        for conflict_node in conflict_nodes:
                            # Check if removing conflict_node and adding candidate improves
                            # New set would have same size, so we need to check if it enables more additions
                            # This is complex; skip for now
                            pass
                if improved:
                    break

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    if not best_set:
        # Fallback: return weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    result = candidates[list(best_set)]
    
    # Final verification
    if len(result) > 1:
        from scipy.spatial.distance import cdist
        dists = cdist(result, result)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_norm = np.max(np.linalg.norm(result, axis=1))
        
        if min_dist < max_norm - 1e-9:
            # Fallback to weight-2 vectors
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
