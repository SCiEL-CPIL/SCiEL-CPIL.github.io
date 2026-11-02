import numpy as np
import itertools
import time

D = 11

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

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Add-focused simulated annealing with shared conflict-count array."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

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
            v = min_cands[rng.integers(len(min_cands))]

            k = min_count
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
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

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

    # Parity-based decomposition: vectors with odd sum of coordinates only conflict
    # with other odd-sum vectors (dot product of two odd-parity vectors is odd,
    # so |dot| >= 3 > 2 always causes conflict). Even-parity vectors similarly
    # only conflict with even-parity vectors. This makes the conflict graph bipartite
    # within each parity class, enabling independent optimization.
    
    # Compute parity of each W4 vector
    coord_sums = vectors_v4.sum(axis=1)
    parity = np.abs(coord_sums) % 2  # 0 = even, 1 = odd
    
    # Build separate conflict matrices for each parity class
    odd_mask = (parity == 1)
    even_mask = (parity == 0)
    
    odd_indices = np.where(odd_mask)[0]
    even_indices = np.where(even_mask)[0]
    
    N_odd = len(odd_indices)
    N_even = len(even_indices)
    
    # Conflict submatrices
    if N_odd > 0:
        conflicts_odd = conflicts[np.ix_(odd_indices, odd_indices)]
    else:
        conflicts_odd = np.zeros((0, 0), dtype=bool)
    
    if N_even > 0:
        conflicts_even = conflicts[np.ix_(even_indices, even_indices)]
    else:
        conflicts_even = np.zeros((0, 0), dtype=bool)
    
    def greedy_min_degree_sub(conflicts_sub, N_sub):
        """Greedy independent set on a sub-conflict matrix."""
        if N_sub == 0:
            return np.zeros(0, dtype=bool)
        
        in_set = np.zeros(N_sub, dtype=bool)
        alive = np.ones(N_sub, dtype=bool)
        
        # Precompute neighbor lists for this subgraph
        neighbors_sub = [np.where(conflicts_sub[i])[0] for i in range(N_sub)]
        neighbor_counts_sub = np.array([len(n) for n in neighbors_sub], dtype=np.int32)
        degrees = neighbor_counts_sub.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sub[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts_sub[rem][:, to_remove].sum(axis=1)
        
        return in_set
    
    def sa_local_search_sub(in_set_init, conflicts_sub, N_sub, max_time=10.0):
        """SA local search on a sub-conflict matrix."""
        if N_sub == 0:
            return in_set_init, 0
        
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        neighbors_sub = [np.where(conflicts_sub[i])[0] for i in range(N_sub)]
        
        count_conflicts = np.zeros(N_sub, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sub[i]] += 1
        
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
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
            
            if accept:
                to_remove = np.where(conflicts_sub[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors_sub[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors_sub[v]] += 1
                
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
    
    # Optimize each parity class independently
    best_odd = np.zeros(N_odd, dtype=bool)
    best_even = np.zeros(N_even, dtype=bool)
    
    n_restarts = 0
    max_restarts = 80  # More restarts since subproblems are smaller
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        # Allocate time between odd and even subproblems
        sa_time_odd = min(remaining_time * 0.5, 8.0)
        sa_time_even = min(remaining_time * 0.5, 8.0)
        
        # Optimize odd parity class
        if N_odd > 0:
            odd_init = greedy_min_degree_sub(conflicts_odd, N_odd)
            odd_result, odd_size = sa_local_search_sub(odd_init, conflicts_odd, N_odd, max_time=sa_time_odd)
            if odd_size > best_odd.sum():
                best_odd = odd_result.copy()
        
        # Optimize even parity class
        if N_even > 0:
            even_init = greedy_min_degree_sub(conflicts_even, N_even)
            even_result, even_size = sa_local_search_sub(even_init, conflicts_even, N_even, max_time=sa_time_even)
            if even_size > best_even.sum():
                best_even = even_result.copy()
        
        # Polish: second pass on best found
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 2:
                if N_odd > 0:
                    odd_polish, odd_polish_size = sa_local_search_sub(best_odd, conflicts_odd, N_odd, max_time=polish_time)
                    if odd_polish_size > best_odd.sum():
                        best_odd = odd_polish.copy()
                if N_even > 0:
                    even_polish, even_polish_size = sa_local_search_sub(best_even, conflicts_even, N_even, max_time=polish_time)
                    if even_polish_size > best_even.sum():
                        best_even = even_polish.copy()
    
    # Combine results from both parity classes
    best_in_set = np.zeros(N4, dtype=bool)
    best_in_set[odd_indices] = best_odd
    best_in_set[even_indices] = best_even
    best_size = best_in_set.sum()

    # Final zero-conflict polish: add any remaining W4 vectors that don't conflict
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 0:
        # Rebuild conflict count for the best set
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1

        # Greedily add any non-conflicting vertices
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            not_in_set = np.where(~best_in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conflict = not_in_set[conflict_count[not_in_set] == 0]
            if len(zero_conflict) > 0:
                for v in zero_conflict:
                    best_in_set[v] = True
                    for u in neighbors[v]:
                        if not best_in_set[u]:
                            conflict_count[u] += 1
                improved = True

        # Try 1-for-1 swaps to improve (within parity classes only)
        if time.time() < deadline - 1:
            for _ in range(300):
                if time.time() >= deadline - 1:
                    break
                in_set_indices = np.where(best_in_set)[0]
                not_in_set = np.where(~best_in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    break
                v = in_set_indices[rng.integers(len(in_set_indices))]
                v_parity = parity[v]
                # Only consider swaps within same parity class
                valid_u = not_in_set[(parity[not_in_set] == v_parity) & (~conflicts[v, not_in_set])]
                if len(valid_u) == 0:
                    continue
                u = valid_u[rng.integers(len(valid_u))]
                u_conflicts = np.where(conflicts[u] & best_in_set)[0]
                u_conflicts = u_conflicts[u_conflicts != v]
                if len(u_conflicts) == 0:
                    # Clean swap
                    best_in_set[v] = False
                    for w in neighbors[v]:
                        if not best_in_set[w]:
                            conflict_count[w] -= 1
                    best_in_set[u] = True
                    for w in neighbors[u]:
                        if not best_in_set[w]:
                            conflict_count[w] += 1

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
