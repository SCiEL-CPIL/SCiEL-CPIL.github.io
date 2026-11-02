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
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_fast(seed_rng=None):
        """Faster greedy: min-degree with random tie-breaking, no expensive second-neighbor score."""
        if seed_rng is None:
            seed_rng = rng
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            
            # Fast tie-break: random among min-degree candidates
            chosen = cands[seed_rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.concatenate(([chosen], nbrs))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized degree update
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def polish_greedy(in_set, max_time=5.0):
        """Batch greedy polish: add all mutually non-conflicting addable vertices iteratively."""
        polish = in_set.copy()
        polish_size = polish.sum()
        polish_deadline = time.time() + max_time
        
        # Precompute float version for matrix mult
        polish_float = polish.astype(np.float64)
        
        changed = True
        while changed and time.time() < polish_deadline:
            changed = False
            not_in = np.where(~polish)[0]
            if len(not_in) == 0:
                break
            
            # Find all vertices that don't conflict with current set
            conflict_counts = conflicts[not_in] @ polish_float
            addable = not_in[conflict_counts == 0]
            
            if len(addable) == 0:
                break
            
            # Among addable, find a maximal independent set greedily
            # Sort by degree (ascending) for better packing
            addable_degrees = neighbor_counts[addable]
            order = np.argsort(addable_degrees)
            sorted_addable = addable[order]
            
            added_this_round = []
            for v in sorted_addable:
                if time.time() >= polish_deadline:
                    break
                # Check if v conflicts with any already-added-in-this-round vertex
                if len(added_this_round) == 0:
                    conflict_with_added = False
                else:
                    # Check against added_this_round
                    conflict_with_added = np.any(conflicts[v, added_this_round])
                
                if not conflict_with_added:
                    polish[v] = True
                    polish_float[v] = 1.0
                    added_this_round.append(v)
                    polish_size += 1
                    changed = True
            
            if len(added_this_round) == 0:
                break
        
        return polish, polish_size
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Efficient SA with add/remove/swap moves using precomputed structures."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55 and len(not_in_set) > 0:
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
                    
            elif move_type < 0.75 and len(in_set_indices) > 0:
                # REMOVE move
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
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    # Prefer not-in-set vertices with low conflict count
                    counts_not_in = count_conflicts[not_in_set]
                    min_c = np.min(counts_not_in)
                    min_cands_not_in = not_in_set[counts_not_in == min_c]
                    v_add = min_cands_not_in[rng.integers(len(min_cands_not_in))]
                    
                    # Check if v_add conflicts with any in-set vertex other than v_remove
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    
                    if len(other_conflicts) == 0:
                        # Clean swap
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        # Complex swap
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
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200  # More restarts with shorter SA each
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        elapsed = time.time() - start_time
        
        # 1. Greedy init with varied random seed
        restart_rng = np.random.default_rng(42 + n_restarts * 7919)
        in_set = greedy_fast(restart_rng)
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Polish: quick greedy add
        remaining = deadline - 8 - time.time()
        if remaining > 2:
            polished, polished_size = polish_greedy(in_set, max_time=min(remaining * 0.15, 3.0))
            if polished_size > current_size:
                in_set = polished
                current_size = polished_size
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
        
        # 3. SA Local Search
        remaining = deadline - 8 - time.time()
        if remaining > 2:
            # Allocate more time to SA as we go, but cap per-restart
            sa_time = min(remaining * 0.7, 12.0)
            if sa_time < 1.5:
                sa_time = 1.5
                
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()
                in_set = sa_result
                current_size = sa_size
        
        # 4. Final polish after SA
        remaining = deadline - 5 - time.time()
        if remaining > 2 and current_size == best_size:
            polished, polished_size = polish_greedy(in_set, max_time=min(remaining * 0.3, 4.0))
            if polished_size > best_size:
                best_size = polished_size
                best_in_set = polished.copy()
        
        # Early exit if we've been at the same best for too long
        if n_restarts > 20 and time.time() - start_time > 60:
            # Check if we're making progress
            pass
    
    # Final result - skip expensive verification since conflict matrix already ensures correctness
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
