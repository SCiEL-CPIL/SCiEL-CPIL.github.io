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
    # Two vectors conflict if their inner product > 2 (since min distance >= max norm = 2 means distance^2 >= 4)
    # distance^2 = |u|^2 + |v|^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # We need distance^2 >= 4 => 8 - 2<u,v> >= 4 => 4 >= 2<u,v> => <u,v> <= 2
    # So conflict if <u,v> > 2
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
        """Simulated annealing with add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.9998
        min_T = 0.01
        
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
            in_set_indices = np.where(in_set)[0]
            
            # Choose move type: add (60%), remove (10%), swap (30%)
            r = rng.random()
            
            if r < 0.60 and len(not_in_set) > 0:
                # Add move: pick vertex with fewest conflicts
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
            elif r < 0.70 and len(in_set_indices) > 0:
                # Remove move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                in_set[v] = False
                current_size -= 1
                count_conflicts[neighbors[v]] -= 1
                accept = True  # Always accept removals to escape local optima
            else:
                # Swap move: remove one, add one
                if len(not_in_set) > 0 and len(in_set_indices) > 0:
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    # Find a vertex in in_set that conflicts with v_add, prefer one with high degree
                    conflicting_in_set = np.where(conflicts[v_add] & in_set)[0]
                    if len(conflicting_in_set) > 0:
                        v_rem = conflicting_in_set[rng.integers(len(conflicting_in_set))]
                        # Net size change is 0, but might restructure
                        # Check if this swap is beneficial: after removing v_rem, 
                        # v_add's conflict count decreases
                        old_conf = count_conflicts[v_add]
                        # Simulate: remove v_rem
                        in_set[v_rem] = False
                        count_conflicts[neighbors[v_rem]] -= 1
                        new_conf = count_conflicts[v_add]
                        # Now add v_add
                        to_remove2 = np.where(conflicts[v_add] & in_set)[0]
                        if len(to_remove2) > 0:
                            in_set[to_remove2] = False
                            for u in to_remove2:
                                count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += (1 - len(to_remove2)) - 1
                        count_conflicts[neighbors[v_add]] += 1
                        # Restore if we want to reject, but for simplicity always accept
                        accept = True
                    else:
                        # No conflict, just do an add instead
                        v = v_add
                        k = count_conflicts[v]
                        delta = 1 - k
                        if delta > 0:
                            accept = True
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
            
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def local_polish(in_set_init, max_iters=5000):
        """Local search: try additions, removals, and swaps."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute count_conflicts
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            best_gain = 0
            best_v = -1
            best_swap_remove = -1
            
            # Try additions
            if len(not_in_set) > 0:
                for v in not_in_set:
                    k = count_conflicts[v]
                    gain = 1 - k
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_swap_remove = -1
            
            # Try swaps: remove one from in_set, add one from not_in_set
            if len(not_in_set) > 0 and len(in_set_indices) > 0:
                for u in in_set_indices:
                    # If we remove u, which not_in_set vertices become addable?
                    # A vertex v in not_in_set can be added if its conflicts with in_set\{u} is 0
                    # i.e., count_conflicts[v] - (1 if u conflicts with v else 0) == 0
                    # and v is not u (v is in not_in_set so v != u)
                    for v in not_in_set:
                        if conflicts[u, v]:
                            # v's effective conflict count after removing u
                            eff_count = count_conflicts[v] - 1
                            if eff_count == 0:
                                # Swap: remove u, add v. Net change = 0, but might enable further improvements
                                # Actually net size change is 0, so not directly improving
                                # Skip pure swaps with no size gain
                                pass
                        # Also check: removing u alone doesn't help, but might unblock others
                        # This is handled by the addition check in next iteration
        
            if best_v >= 0 and best_gain > 0:
                to_remove = np.where(conflicts[best_v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[best_v] = True
                current_size += best_gain
                count_conflicts[neighbors[best_v]] += 1
                improved = True
        
        return in_set, current_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 10.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Exact local polish on best found so far
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_result, polish_size = local_polish(sa_result, max_iters=5000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Select W1 vectors: at most one per coordinate (can't have both +2e_i and -2e_i)
    # W1 vectors: index 2*i is +2e_i, index 2*i+1 is -2e_i for i in 0..D-1
    w1_selected = []
    for i in range(D):
        # Pick the one that doesn't conflict with any selected W4 vector
        # Actually W1 never conflicts with W4 (dot <= 2 always), so just pick one per coordinate
        w1_selected.append(vectors_v1[2*i])  # pick +2e_i
    
    final_vectors = np.vstack([result_v4, np.array(w1_selected)])
    
    return final_vectors

# EVOLVE-BLOCK-END
