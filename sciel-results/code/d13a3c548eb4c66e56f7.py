"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Generate W4 supports (weight 4, entries in {-1,0,1}, norm^2=4)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Build conflict graph on supports
    # Two supports conflict if they share >= 3 positions
    conflict_matrix = np.zeros((n_supports, n_supports), dtype=bool)
    
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                conflict_matrix[i, j] = True
                conflict_matrix[j, i] = True
    
    # Precompute neighbors for faster operations
    neighbors = [np.where(conflict_matrix[i])[0] for i in range(n_supports)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    best_size = 0
    best_in_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def greedy_max_coverage():
        """Greedy independent set prioritizing nodes that remove fewest future candidates."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # For each candidate, count how many alive nodes it would remove
            scores = np.zeros(len(alive_indices), dtype=np.int32)
            for idx, node in enumerate(alive_indices):
                nbrs = neighbors[node]
                alive_nbrs = np.intersect1d(nbrs, alive_indices)
                scores[idx] = len(alive_nbrs)
            
            min_score = np.min(scores)
            cands = alive_indices[scores == min_score]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0, seed=None):
        """Simulated annealing with multiple move types."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
            
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.9998
        min_T = 0.01
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Single insert move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                s = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflict_matrix[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors[s]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.8:
                # Swap move: remove 1, add 1
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                    
                # Sample a few candidates for removal
                n_sample = min(3, len(in_set_indices))
                sample_idx = rng.choice(len(in_set_indices), size=n_sample, replace=False)
                sample_remove = in_set_indices[sample_idx]
                
                swap_accepted = False
                for r in sample_remove:
                    # Find non-conflicting nodes not in set
                    non_conf = np.where(~conflict_matrix[r])[0]
                    candidates = np.intersect1d(non_conf, not_in_set)
                    
                    if len(candidates) == 0:
                        continue
                    
                    # Pick a random candidate
                    s = candidates[rng.integers(len(candidates))]
                    
                    # Check if s conflicts with any other in-set node
                    s_conflicts = np.where(conflict_matrix[s] & in_set)[0]
                    s_conflicts = s_conflicts[s_conflicts != r]  # exclude r
                    
                    if len(s_conflicts) > 0:
                        continue  # Can't do simple swap
                    
                    # Perform swap: remove r, add s
                    in_set[r] = False
                    count_conflicts[neighbors[r]] -= 1
                    
                    in_set[s] = True
                    count_conflicts[neighbors[s]] += 1
                    
                    # Size unchanged, so it's a neutral move
                    # Accept with probability 0.3 to allow exploration
                    if rng.random() < 0.3:
                        swap_accepted = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        break
                    else:
                        # Revert
                        in_set[r] = True
                        count_conflicts[neighbors[r]] += 1
                        in_set[s] = False
                        count_conflicts[neighbors[s]] -= 1
                        stuck_counter += 1
                
                if not swap_accepted:
                    stuck_counter += 1
                    
            else:
                # 2-for-1 swap: remove 2 conflicting nodes, add 1 new node
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Sample a few candidates for the new node
                n_sample = min(5, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_nodes = not_in_set[sample_idx]
                
                swap_accepted = False
                for s in sample_nodes:
                    # Get conflicts of s that are currently in the set
                    conflicts_in_set = np.where(conflict_matrix[s] & in_set)[0]
                    k = len(conflicts_in_set)
                    
                    if k < 2:
                        continue  # Need at least 2 to remove for a 2-for-1 swap
                    
                    delta = 1 - k
                    
                    if delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        # Perform the swap
                        to_remove = conflicts_in_set
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        
                        in_set[s] = True
                        current_size += delta
                        count_conflicts[neighbors[s]] += 1
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        
                        swap_accepted = True
                        break
                
                if not swap_accepted:
                    stuck_counter += 1
            
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with different strategies
    n_restarts = 0
    max_restarts = 500
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose initialization strategy
        if n_restarts % 3 == 0:
            in_set = greedy_min_degree()
        else:
            in_set = greedy_max_coverage()
            
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        seed = n_restarts * 1000 + 42
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time, seed=seed)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 3:
                polish_seed = n_restarts * 1000 + 43
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time, seed=polish_seed)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Build final vector list from selected supports
    selected_support_indices = np.where(best_in_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        # Generate all 16 sign variants for this support
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            final_vectors.append(v)
    
    # Add W1 vectors: one per position (22 total, but we can only pick one sign per position)
    # Actually, we can include both signs at different positions, but not both at the same position
    # Let's include all 22 (both signs for each position) - wait, that would conflict
    # Two W1 vectors at the same position with opposite signs: dot = -4 <= 2, OK
    # Two W1 vectors at the same position with same sign: dot = 4 > 2, CONFLICT
    # So we can include at most one W1 vector per position
    # Let's include one per position (22 total)
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        final_vectors.append(v)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    # Fallback: if something went wrong, use deterministic construction
    if len(result) == 0 or best_size == 0:
        # Deterministic fallback: pick non-conflicting supports greedily
        selected_supports = []
        for s in supports:
            valid = True
            for t in selected_supports:
                intersection = set(s).intersection(t)
                if len(intersection) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        # Add W1 vectors
        for pos in range(D):
            v = [0] * D
            v[pos] = 2
            fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
