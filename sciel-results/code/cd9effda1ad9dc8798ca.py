"""Kissing number in dimension 11 - Improved Local Search with Grouped Swaps."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates: norm squared 4
    # Type A: weight 4, entries in {-1, 1}
    # Type B: weight 1, entries in {-2, 2}
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of lists for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy min-degree initialization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_grouped(in_set, max_time=10.0):
        """Local search using 1-2 and 1-3 swaps grouped by unique conflict."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    continue
            
            # 2. Grouped Swap Search
            # Find vertices not in set with conflict_count == 1
            # These are "blocked" by exactly one vertex in the set
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                one_conf = not_in_set[counts == 1]
                
                if len(one_conf) > 0:
                    # Group these vertices by their unique blocker
                    # unique_blocker[v] is the vertex in in_set that conflicts with v
                    groups = {}
                    for v in one_conf:
                        # Find the unique neighbor in in_set
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 1:
                            u = nbrs_in_set[0]
                            if u not in groups:
                                groups[u] = []
                            groups[u].append(v)
                    
                    # Check for 1-2 swaps (remove u, add v1, v2)
                    # v1, v2 must not conflict with each other
                    swap_found = False
                    for u, vs in groups.items():
                        if len(vs) >= 2:
                            # Check pairs in vs
                            # Optimization: if len(vs) is large, check random pairs or all?
                            # For efficiency, check all pairs if len < 10, else sample
                            if len(vs) < 10:
                                for i in range(len(vs)):
                                    for j in range(i+1, len(vs)):
                                        v1, v2 = vs[i], vs[j]
                                        if not conflicts[v1, v2]:
                                            # Perform swap
                                            in_set[u] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            
                                            # Update conflict counts
                                            conflict_count[neighbors[u]] -= 1
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            
                                            current_size += 1
                                            improved = True
                                            swap_found = True
                                            
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            break
                                    if swap_found:
                                        break
                            else:
                                # Sample pairs
                                for _ in range(10):
                                    i = rng.integers(len(vs))
                                    j = rng.integers(len(vs))
                                    if i == j: continue
                                    v1, v2 = vs[i], vs[j]
                                    if not conflicts[v1, v2]:
                                        in_set[u] = False
                                        in_set[v1] = True
                                        in_set[v2] = True
                                        
                                        conflict_count[neighbors[u]] -= 1
                                        conflict_count[neighbors[v1]] += 1
                                        conflict_count[neighbors[v2]] += 1
                                        
                                        current_size += 1
                                        improved = True
                                        swap_found = True
                                        
                                        if current_size > best_size:
                                            best_size = current_size
                                            best_in_set = in_set.copy()
                                        break
                            if swap_found:
                                break
                    
                    if swap_found:
                        stuck = 0
                        continue
            
            # 3. Check for 1-3 swaps (remove u, add v1, v2, v3)
            # This is similar but requires 3 non-conflicting vertices in a group
            # We can skip this for now if 1-2 is frequent, or add it if stuck
            if not improved and stuck > 5:
                 # Re-evaluate groups for 1-3 swaps
                 # This is more expensive, so only do it when stuck
                 not_in_set = np.where(~in_set)[0]
                 if len(not_in_set) > 0:
                     counts = conflict_count[not_in_set]
                     one_conf = not_in_set[counts == 1]
                     if len(one_conf) > 0:
                         groups = {}
                         for v in one_conf:
                             nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                             if len(nbrs_in_set) == 1:
                                 u = nbrs_in_set[0]
                                 if u not in groups:
                                     groups[u] = []
                                 groups[u].append(v)
                         
                         swap_found = False
                         for u, vs in groups.items():
                             if len(vs) >= 3:
                                 # Find 3 non-conflicting vertices in vs
                                 # Simple greedy check
                                 chosen_vs = []
                                 for v in vs:
                                     if all(not conflicts[v, cv] for cv in chosen_vs):
                                         chosen_vs.append(v)
                                         if len(chosen_vs) == 3:
                                             break
                                 if len(chosen_vs) == 3:
                                     in_set[u] = False
                                     for v in chosen_vs:
                                         in_set[v] = True
                                     conflict_count[neighbors[u]] -= 1
                                     for v in chosen_vs:
                                         conflict_count[neighbors[v]] += 1
                                     
                                     current_size += 2 # Remove 1, Add 3 -> Net +2
                                     improved = True
                                     swap_found = True
                                     
                                     if current_size > best_size:
                                         best_size = current_size
                                         best_in_set = in_set.copy()
                                     break
                                 if swap_found:
                                     break
                         if swap_found:
                             stuck = 0
                             continue

            if not improved:
                stuck += 1
                if stuck > 20:
                    # Random perturbation: remove a random vertex from set
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Initial Greedy
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on initial
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.4, 15.0)
        ls_result, ls_size = local_search_grouped(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # Restarts
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.3, 8.0)
        ls_result, ls_size = local_search_grouped(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                # Update counts for other not-in-set vertices
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
