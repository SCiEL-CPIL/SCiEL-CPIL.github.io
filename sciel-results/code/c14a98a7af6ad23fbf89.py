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

    # Group W4 vectors by their support (tuple of nonzero positions)
    # For a given support, all 16 sign variants are mutually compatible
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
    
    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    
    # Build support-level conflict matrix:
    # Two supports conflict if they share >= 3 positions
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True
    
    # Each support contributes 16 vectors if selected
    vectors_per_support = 16
    
    # W1 vectors are compatible with all W4 vectors
    # W1 vector at p with value +/-2. W4 has +/-1 at 4 positions.
    # Dot product is +/-1 if p in support, else 0.
    # Max abs dot is 1 <= 2. So no conflicts.
    
    # Now solve independent set on support graph
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree_supports():
        """Greedy independent set with min-degree heuristic on support graph."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search_supports(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing on support graph with efficient conflict updates."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
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
                # Efficiently update conflict counts
                # First, find which neighbors of s are currently in the set (to be removed)
                to_remove = np.where(support_conflicts[s] & in_set)[0]
                
                if len(to_remove) > 0:
                    # For each removed vertex u, decrement conflict count of all neighbors of u
                    # that are still in the set (excluding s itself since we're adding it)
                    for u in to_remove:
                        in_set[u] = False
                        # Decrement count_conflicts for neighbors of u that are in in_set
                        nbrs_u = neighbors_sup[u]
                        mask = in_set[nbrs_u]
                        if mask.any():
                            count_conflicts[nbrs_u[mask]] -= 1
                
                # Add s to the set
                in_set[s] = True
                current_size += delta
                
                # Increment conflict count for all neighbors of s that are in the set
                nbrs_s = neighbors_sup[s]
                mask_s = in_set[nbrs_s]
                if mask_s.any():
                    count_conflicts[nbrs_s[mask_s]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 500  # More restarts possible due to smaller problem

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree_supports()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # 3. Polish: try to add any free support (only if SA improved things)
        polish_deadline = deadline - 2
        if time.time() < polish_deadline and sa_size > best_size - 1:
             polish_in_set = best_support_set.copy()
             count_conflicts = np.zeros(n_supports, dtype=np.int32)
             for i in np.where(polish_in_set)[0]:
                 count_conflicts[neighbors_sup[i]] += 1
             
             improved = True
             while improved and time.time() < polish_deadline:
                 improved = False
                 not_in_set = np.where(~polish_in_set)[0]
                 for v in not_in_set:
                     if time.time() > polish_deadline:
                         break
                     if count_conflicts[v] == 0:
                         polish_in_set[v] = True
                         count_conflicts[neighbors_sup[v]] += 1
                         improved = True
                         
             if polish_in_set.sum() > best_size:
                 best_size = polish_in_set.sum()
                 best_support_set = polish_in_set.copy()

    # Final result construction with additional polish pass
    # Try to add any free supports one more time with a more thorough approach
    polish_in_set = best_support_set.copy()
    count_conflicts = np.zeros(n_supports, dtype=np.int32)
    for i in np.where(polish_in_set)[0]:
        count_conflicts[neighbors_sup[i]] += 1
    
    # Aggressive polish: repeatedly try to add vertices with 0 conflicts
    improved = True
    polish_deadline_final = deadline - 1
    while improved and time.time() < polish_deadline_final:
        improved = False
        not_in_set = np.where(~polish_in_set)[0]
        for v in not_in_set:
            if time.time() > polish_deadline_final:
                break
            if count_conflicts[v] == 0:
                polish_in_set[v] = True
                count_conflicts[neighbors_sup[v]] += 1
                improved = True
    
    if polish_in_set.sum() > best_size:
        best_size = polish_in_set.sum()
        best_support_set = polish_in_set.copy()
    
    final_v4_indices = []
    for s_idx in np.where(best_support_set)[0]:
        final_v4_indices.extend(support_to_indices[supports[s_idx]])
    
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
