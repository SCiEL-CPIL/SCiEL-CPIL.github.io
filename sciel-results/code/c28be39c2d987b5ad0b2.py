"""Kissing number in dimension 11 - maximum independent set on conflict graph."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    
    vectors_v4 = np.array(v4_list, dtype=np.int64)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict matrix: conflict if |dot| > 2 (i.e., dot >= 3 or dot <= -3)
    # For vectors with norm^2=4, distance^2 = 8 - 2*dot
    # We need distance >= norm = 2, so distance^2 >= 4, so 8 - 2*dot >= 4, dot <= 2
    # But we also need to consider the absolute value since we want min pairwise distance
    # Actually, for the kissing configuration, we need |v_i - v_j| >= |v_i| = 2
    # So |v_i - v_j|^2 >= 4
    # |v_i - v_j|^2 = |v_i|^2 + |v_j|^2 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # So 8 - 2*v_i.v_j >= 4 => v_i.v_j <= 2
    # Conflict when v_i.v_j > 2 (i.e., dot >= 3)
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as lists of arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_init(seed=None, strategy="min_degree"):
        """Greedy independent set with different strategies."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
            
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            if strategy == "min_degree":
                degs = degrees[alive_indices]
                mins = np.min(degs)
                cands = alive_indices[degs == mins]
                chosen = cands[local_rng.integers(len(cands))]
            elif strategy == "random":
                chosen = alive_indices[local_rng.integers(len(alive_indices))]
            elif strategy == "low_degree_random":
                # Pick from bottom 20% degree vertices
                degs = degrees[alive_indices]
                threshold = np.percentile(degs, 20)
                cands = alive_indices[degs <= threshold]
                chosen = cands[local_rng.integers(len(cands))]
            else:
                chosen = alive_indices[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized degree update
                degree_drops = conflicts[rem][:, to_remove].sum(axis=1)
                degrees[rem] -= degree_drops
        
        return in_set
    
    def sa_local_search(in_set_init, max_time=5.0, T_start=1.5, T_end=0.05):
        """Simulated annealing with efficient moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()
        
        # Initialize conflict counts
        conflict_counts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        if len(in_set_indices) > 0:
            for i in in_set_indices:
                conflict_counts[neighbors[i]] += 1
        
        T = T_start
        sa_deadline = time.time() + max_time
        stuck = 0
        step = 0
        
        while time.time() < sa_deadline:
            step += 1
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.50 and len(not_in_set) > 0:
                # ADD move: add vertex with fewest conflicts
                counts = conflict_counts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        conflict_counts[neighbors[to_remove]] -= 1  # This is wrong, need to be careful
                        # Actually, for each removed vertex u, we need to decrement conflict_counts for all neighbors of u
                        # But conflict_counts[v] = number of in_set vertices that conflict with v
                        # So when we remove u from in_set, for each neighbor w of u, conflict_counts[w] decreases by 1
                        # But we already set in_set[to_remove] = False, so let's redo this properly
                        for u in to_remove:
                            conflict_counts[neighbors[u]] -= 1
                    in_set[v] = True
                    conflict_counts[neighbors[v]] += 1
                    current_size += delta
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1
                    
            elif move_type < 0.65 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    conflict_counts[neighbors[v]] -= 1
                    current_size += delta
                    stuck += 1
                else:
                    stuck += 1
                    
            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one, add one
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Find good candidates to add
                # Temporarily pretend v_remove is not in set
                temp_counts = conflict_counts.copy()
                temp_counts[neighbors[v_remove]] -= 1
                temp_counts[v_remove] = 0
                
                cands = not_in_set[temp_counts[not_in_set] <= temp_counts[not_in_set].min() + 1]
                if len(cands) == 0:
                    cands = not_in_set
                v_add = cands[rng.integers(len(cands))]
                
                # Compute delta for this swap
                conflicts_with_in_set = np.sum(conflicts[v_add] & in_set)
                # v_remove is in set, so it's counted above
                # After removing v_remove, conflicts_with_in_set decreases by 1 if v_add conflicts with v_remove
                if conflicts[v_add, v_remove]:
                    conflicts_after_remove = conflicts_with_in_set - 1
                else:
                    conflicts_after_remove = conflicts_with_in_set
                
                delta = 1 - int(conflicts_after_remove)
                
                if delta >= 0 or rng.random() < np.exp(delta / T):
                    # Execute swap
                    to_remove = np.where(conflicts[v_add] & in_set)[0]
                    in_set[to_remove] = False
                    for u in to_remove:
                        conflict_counts[neighbors[u]] -= 1
                    in_set[v_add] = True
                    conflict_counts[neighbors[v_add]] += 1
                    current_size += delta
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1
                    
            else:
                # 2-SWAP: remove 2, add 2
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 == v2:
                        v2 = in_set_indices[rng.integers(len(in_set_indices))]
                        if v1 == v2:
                            stuck += 1
                        else:
                            # Find 2 good candidates
                            temp_counts = conflict_counts.copy()
                            for v_rem in [v1, v2]:
                                temp_counts[neighbors[v_rem]] -= 1
                                temp_counts[v_rem] = 0
                            
                            cands = not_in_set[temp_counts[not_in_set] <= temp_counts[not_in_set].min() + 2]
                            if len(cands) < 2:
                                cands = not_in_set
                            
                            # Pick 2 candidates
                            sample_size = min(2, len(cands))
                            sample_idx = rng.choice(len(cands), size=sample_size, replace=False)
                            found = cands[sample_idx]
                            
                            if len(found) == 2:
                                # Check if they're compatible with each other and with remaining set
                                v_a, v_b = found
                                if not conflicts[v_a, v_b]:
                                    # Check conflicts with remaining in_set (excluding v1, v2)
                                    remaining_in_set = in_set.copy()
                                    remaining_in_set[v1] = False
                                    remaining_in_set[v2] = False
                                    
                                    conf_a = np.sum(conflicts[v_a] & remaining_in_set)
                                    conf_b = np.sum(conflicts[v_b] & remaining_in_set)
                                    
                                    delta = 2 - int(conf_a) - int(conf_b)
                                    
                                    if delta >= 0 or rng.random() < np.exp(delta / T):
                                        to_remove = np.where(conflicts[v_a] & in_set)[0]
                                        to_remove = np.union1d(to_remove, np.where(conflicts[v_b] & in_set)[0])
                                        in_set[to_remove] = False
                                        for u in to_remove:
                                            conflict_counts[neighbors[u]] -= 1
                                        in_set[v_a] = True
                                        in_set[v_b] = True
                                        conflict_counts[neighbors[v_a]] += 1
                                        conflict_counts[neighbors[v_b]] += 1
                                        current_size += delta
                                        
                                        if current_size > best_size:
                                            best_size = current_size
                                            best_in_set = in_set.copy()
                                            stuck = 0
                                        else:
                                            stuck += 1
                                    else:
                                        stuck += 1
                                else:
                                    stuck += 1
                            else:
                                stuck += 1
                    else:
                        # Find 2 good candidates
                        temp_counts = conflict_counts.copy()
                        for v_rem in [v1, v2]:
                            temp_counts[neighbors[v_rem]] -= 1
                            temp_counts[v_rem] = 0
                        
                        cands = not_in_set[temp_counts[not_in_set] <= temp_counts[not_in_set].min() + 2]
                        if len(cands) < 2:
                            cands = not_in_set
                        
                        sample_size = min(2, len(cands))
                        sample_idx = rng.choice(len(cands), size=sample_size, replace=False)
                        found = cands[sample_idx]
                        
                        if len(found) == 2:
                            v_a, v_b = found
                            if not conflicts[v_a, v_b]:
                                remaining_in_set = in_set.copy()
                                remaining_in_set[v1] = False
                                remaining_in_set[v2] = False
                                
                                conf_a = np.sum(conflicts[v_a] & remaining_in_set)
                                conf_b = np.sum(conflicts[v_b] & remaining_in_set)
                                
                                delta = 2 - int(conf_a) - int(conf_b)
                                
                                if delta >= 0 or rng.random() < np.exp(delta / T):
                                    to_remove = np.where(conflicts[v_a] & in_set)[0]
                                    to_remove = np.union1d(to_remove, np.where(conflicts[v_b] & in_set)[0])
                                    in_set[to_remove] = False
                                    for u in to_remove:
                                        conflict_counts[neighbors[u]] -= 1
                                    in_set[v_a] = True
                                    in_set[v_b] = True
                                    conflict_counts[neighbors[v_a]] += 1
                                    conflict_counts[neighbors[v_b]] += 1
                                    current_size += delta
                                    
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                        stuck = 0
                                    else:
                                        stuck += 1
                                else:
                                    stuck += 1
                            else:
                                stuck += 1
                        else:
                            stuck += 1
                else:
                    stuck += 1
            
            # Cooling
            if stuck > 1000:
                T = max(T, 1.0)
                stuck = 0
            
            T = max(T_end, T * 0.9995)
            
            # Early termination if we've been stuck for too long at low T
            if T <= T_end * 1.1 and stuck > 5000:
                break
        
        return best_in_set, best_size
    
    def greedy_polish(in_set_init):
        """Fast greedy: add any vertex that doesn't conflict with current set."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        conflict_counts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            conflict_counts[neighbors[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            addable = not_in[conflict_counts[not_in] == 0]
            
            if len(addable) > 0:
                # Add all addable vertices that don't conflict with each other
                # Simple approach: add one at a time
                v_idx = addable[0]
                in_set[v_idx] = True
                current_size += 1
                conflict_counts[neighbors[v_idx]] += 1
                changed = True
        
        return in_set, current_size
    
    # Luby-like restart pattern
    luby_pattern = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    
    n_restarts = 0
    max_restarts = 300
    luby_idx = 0
    
    strategies = ["min_degree", "random", "low_degree_random"]
    
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        
        luby_len = luby_pattern[luby_idx % len(luby_pattern)]
        luby_idx += 1
        
        # 1. Greedy Init with different seeds and strategies
        seed = n_restarts * 7 + 13
        strategy = strategies[n_restarts % len(strategies)]
        in_set = greedy_init(seed=seed, strategy=strategy)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 12 - time.time()
        if remaining_time < 3:
            break
        
        base_sa_time = min(remaining_time * 0.4, 8.0)
        sa_time = base_sa_time * (luby_len / 4.0)
        sa_time = max(sa_time, 1.0)
        sa_time = min(sa_time, remaining_time * 0.6)
        
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Greedy polish
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2:
            polish_set, polish_size = greedy_polish(sa_result if sa_size >= current_size else in_set)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_set.copy()
        
        # 4. If we have time, do a second SA on the best found
        remaining_time = deadline - 6 - time.time()
        if remaining_time > 8 and n_restarts % 4 == 0:
            polish_time = min(remaining_time * 0.5, 6.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(best_in_set, max_time=polish_time, T_start=0.8, T_end=0.02)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final high-precision polish
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 3:
        hp_time = min(remaining_time * 0.8, 5.0)
        if hp_time > 1:
            hp_result, hp_size = sa_local_search(best_in_set, max_time=hp_time, T_start=0.3, T_end=0.01)
            if hp_size > best_size:
                best_size = hp_size
                best_in_set = hp_result
    
    # Final greedy polish
    if time.time() < deadline - 3:
        final_polish, final_size = greedy_polish(best_in_set)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
