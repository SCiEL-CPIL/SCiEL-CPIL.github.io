"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Check if graph is bipartite by checking for odd cycles
    # For W4 vectors: two vectors conflict if their dot product > 2
    # Let's check the structure more carefully
    # Actually, let's just use a good heuristic approach
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_weighted_init():
        """Greedy independent set with weighted scoring: prefer low-degree, then prefer vertices with many non-neighbors."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
        # Compute "non-neighbor count" for each vertex (number of vertices it doesn't conflict with)
        non_neighbor_counts = np.array([N4 - 1 - nc for nc in neighbor_counts], dtype=np.float64)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Score: lower degree is better, higher non-neighbor count is better
            # Combined score: degree - alpha * non_neighbor_count
            # We want to minimize degree, so use degree as primary, non_neighbor as tie-breaker
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            
            if len(cands) > 1:
                # Among min-degree candidates, prefer those with most non-neighbors among alive
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    # Prefer vertices whose neighbors are "expensive" to remove
                    # i.e., neighbors that have high degree
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                    else:
                        scores[idx] = 0
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def local_search_2opt(in_set_init, max_time=10.0):
        """Local search with 1-opt (add/remove) and 2-opt (swap pairs) moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        stuck_counter = 0
        
        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.45 and len(not_in_set) > 0:
                # ADD move: try to add a vertex with fewest conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                # Smart swap: pick v_add that has fewest conflicts, v_remove that's most "redundant"
                counts_add = count_conflicts[not_in_set]
                min_count_add = np.min(counts_add)
                cands_add = not_in_set[counts_add == min_count_add]
                v_add = cands_add[rng.integers(len(cands_add))]
                
                # For v_remove, prefer vertices whose removal frees up the most
                # i.e., vertices that conflict with many other in-set vertices
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                        
            elif move_type < 0.85 and len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                # 2-SWAP: remove 2 in-set, add 2 not-in-set
                idx1, idx2 = rng.choice(len(in_set_indices), size=2, replace=False)
                v_rem1 = in_set_indices[idx1]
                v_rem2 = in_set_indices[idx2]
                
                # Find good candidates to add
                # Create mapping from global index to local index in not_in_set
                map_to_local = -np.ones(N4, dtype=np.int32)
                map_to_local[not_in_set] = np.arange(len(not_in_set))
                
                counts_add = count_conflicts[not_in_set].copy()
                # Temporarily reduce conflicts for v_rem1 and v_rem2
                for u in [v_rem1, v_rem2]:
                    for nb in neighbors[u]:
                        if not in_set[nb]:
                            local_idx = map_to_local[nb]
                            if local_idx != -1:
                                counts_add[local_idx] -= 1
                
                min_count_2 = np.min(counts_add)
                cands_2 = not_in_set[counts_add == min_count_2]
                if len(cands_2) >= 2:
                    idx_a, idx_b = rng.choice(len(cands_2), size=2, replace=False)
                    v_add1 = cands_2[idx_a]
                    v_add2 = cands_2[idx_b]
                    
                    # Check if v_add1 and v_add2 conflict with each other or with remaining in_set
                    if not conflicts[v_add1, v_add2]:
                        # Check conflicts with in_set excluding v_rem1, v_rem2
                        conflicts_1 = np.where(conflicts[v_add1] & in_set)[0]
                        conflicts_2 = np.where(conflicts[v_add2] & in_set)[0]
                        other_conflicts = np.unique(np.concatenate([conflicts_1, conflicts_2]))
                        other_conflicts = other_conflicts[~np.isin(other_conflicts, [v_rem1, v_rem2])]
                        
                        if len(other_conflicts) == 0:
                            delta = 2 - 2  # Net zero, but might be better structurally
                            accept = rng.random() < 0.3 or rng.random() < np.exp(0.5 / T)
                            if accept:
                                in_set[v_rem1] = False
                                in_set[v_rem2] = False
                                in_set[v_add1] = True
                                in_set[v_add2] = True
                                count_conflicts[neighbors[v_rem1]] -= 1
                                count_conflicts[neighbors[v_rem2]] -= 1
                                count_conflicts[neighbors[v_add1]] += 1
                                count_conflicts[neighbors[v_add2]] += 1
                        else:
                            to_remove = np.unique(np.concatenate(([v_rem1, v_rem2], other_conflicts)))
                            delta = 2 - len(to_remove)
                            accept = delta > 0 or (rng.random() < np.exp(delta / T))
                            if accept:
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                in_set[v_add1] = True
                                in_set[v_add2] = True
                                current_size += delta
                                count_conflicts[neighbors[v_add1]] += 1
                                count_conflicts[neighbors[v_add2]] += 1
                            else:
                                stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.92 and len(in_set_indices) >= 1 and len(not_in_set) > 0:
                # 1-REMOVE, 2-ADD: remove 1 in-set, add 2 not-in-set
                v_rem = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Create mapping from global index to local index in not_in_set
                map_to_local = -np.ones(N4, dtype=np.int32)
                map_to_local[not_in_set] = np.arange(len(not_in_set))
                
                counts_add = count_conflicts[not_in_set].copy()
                for nb in neighbors[v_rem]:
                    if not in_set[nb]:
                        local_idx = map_to_local[nb]
                        if local_idx != -1:
                            counts_add[local_idx] -= 1
                
                min_count_2 = np.min(counts_add)
                cands_2 = not_in_set[counts_add == min_count_2]
                if len(cands_2) >= 2:
                    idx_a, idx_b = rng.choice(len(cands_2), size=2, replace=False)
                    v_add1 = cands_2[idx_a]
                    v_add2 = cands_2[idx_b]
                    
                    if not conflicts[v_add1, v_add2]:
                        conflicts_1 = np.where(conflicts[v_add1] & in_set)[0]
                        conflicts_2 = np.where(conflicts[v_add2] & in_set)[0]
                        other_conflicts = np.unique(np.concatenate([conflicts_1, conflicts_2]))
                        other_conflicts = other_conflicts[other_conflicts != v_rem]
                        
                        if len(other_conflicts) == 0:
                            delta = 2 - 1
                            accept = True
                            if accept:
                                in_set[v_rem] = False
                                in_set[v_add1] = True
                                in_set[v_add2] = True
                                count_conflicts[neighbors[v_rem]] -= 1
                                count_conflicts[neighbors[v_add1]] += 1
                                count_conflicts[neighbors[v_add2]] += 1
                                current_size += delta
                        else:
                            to_remove = np.unique(np.concatenate(([v_rem], other_conflicts)))
                            delta = 2 - len(to_remove)
                            accept = delta > 0 or (rng.random() < np.exp(delta / T))
                            if accept:
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                in_set[v_add1] = True
                                in_set[v_add2] = True
                                current_size += delta
                                count_conflicts[neighbors[v_add1]] += 1
                                count_conflicts[neighbors[v_add2]] += 1
                            else:
                                stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # REMOVE move
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    delta = -1
                    accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        in_set[v] = False
                        current_size += delta
                        count_conflicts[neighbors[v]] -= 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1000:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def polish_greedy_add(in_set_init, max_time=5.0):
        """Polish by greedily adding non-conflicting vertices in random order."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        
        for trial in range(8):
            if time.time() >= polish_deadline:
                break
            
            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break
            
            order = rng.permutation(len(not_in))
            
            for idx in order:
                if time.time() >= polish_deadline:
                    break
                
                v = not_in[idx]
                
                if np.any(conflicts[v] & polish_set):
                    continue
                
                polish_set[v] = True
                polish_size += 1
        
        return polish_set, polish_size
    
    # Multiple restarts with diverse initializations
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init (alternate between different heuristics)
        if n_restarts % 3 == 0:
            # Randomized greedy
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
        else:
            in_set = greedy_weighted_init()
        
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        ls_time = min(remaining_time * 0.5, 12.0)
        if ls_time < 2:
            ls_time = 2.0
            
        ls_result, ls_size = local_search_2opt(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # 3. Polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 1:
                polish_result, polish_size = polish_greedy_add(ls_result if ls_size >= current_size else in_set, max_time=polish_time)
                
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                
                # 4. Second local search on polished result
                remaining_time2 = deadline - 3 - time.time()
                if remaining_time2 > 4 and polish_size >= current_size:
                    ls2_time = min(remaining_time2 * 0.6, 8.0)
                    if ls2_time > 2:
                        ls2_result, ls2_size = local_search_2opt(polish_result, max_time=ls2_time)
                        if ls2_size > best_size:
                            best_size = ls2_size
                            best_in_set = ls2_result.copy()
                        
                        # Final polish
                        remaining_time3 = deadline - 2 - time.time()
                        if remaining_time3 > 2:
                            final_polish, final_size = polish_greedy_add(ls2_result, max_time=min(remaining_time3 * 0.5, 4.0))
                            if final_size > best_size:
                                best_size = final_size
                                best_in_set = final_polish.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
