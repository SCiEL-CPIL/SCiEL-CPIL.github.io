"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

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
        """Simulated annealing with add and swap moves."""
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
        
        in_set_indices = np.where(in_set)[0]
        not_in_set_indices = np.where(~in_set)[0]
        
        while time.time() < sa_deadline:
            if len(not_in_set_indices) == 0:
                break
            
            # Choose move type: add (70%) or swap (30%)
            if len(in_set_indices) > 0 and rng.random() < 0.3:
                # Swap: pick v from in_set, u from not_in_set
                v = in_set_indices[rng.integers(len(in_set_indices))]
                u = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                
                # Remove v, add u
                # Net effect: remove v's conflicts with other in_set, add u's conflicts
                # v is in set, u is not. After swap: u in set, v not in set.
                # Conflicts of u with (in_set \ {v}): need to remove those
                nbrs_u_in_set = np.where(conflicts[u] & in_set)[0]
                nbrs_u_in_set = nbrs_u_in_set[nbrs_u_in_set != v]
                
                # If u conflicts with other in_set vertices (besides v), we'd need to remove them too
                # That makes swap complex. For simplicity, only allow swap if u has no conflicts with in_set \ {v}
                if len(nbrs_u_in_set) == 0:
                    # Safe swap
                    in_set[v] = False
                    in_set[u] = True
                    # Update count_conflicts
                    for w in neighbors[v]:
                        count_conflicts[w] -= 1
                    for w in neighbors[u]:
                        count_conflicts[w] += 1
                    # current_size unchanged
                    # Update indices
                    in_set_indices = np.where(in_set)[0]
                    not_in_set_indices = np.where(~in_set)[0]
                    stuck_counter += 1
                else:
                    # Skip this swap, try add instead
                    pass
            else:
                # Add move
                counts = count_conflicts[not_in_set_indices]
                if len(counts) == 0:
                    break
                min_count = np.min(counts)
                min_cands = not_in_set_indices[counts == min_count]
                u = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[u] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for w in to_remove:
                            for x in neighbors[w]:
                                count_conflicts[x] -= 1
                    in_set[u] = True
                    for x in neighbors[u]:
                        count_conflicts[x] += 1
                    current_size += delta
                    in_set_indices = np.where(in_set)[0]
                    not_in_set_indices = np.where(~in_set)[0]
                    
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
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final polish: try all single-vertex additions
    polish_deadline = deadline - 2
    in_set = best_in_set.copy()
    count_conflicts = np.zeros(N4, dtype=np.int32)
    for i in np.where(in_set)[0]:
        count_conflicts[neighbors[i]] += 1
    
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        not_in_set = np.where(~in_set)[0]
        for v in not_in_set:
            if time.time() > polish_deadline:
                break
            k = count_conflicts[v]
            if k == 0:
                # Can add v for free
                in_set[v] = True
                for w in neighbors[v]:
                    count_conflicts[w] += 1
                improved = True
            elif k == 1:
                # Adding v removes 1 vertex, net 0 change. Skip unless it helps future additions.
                pass
        # Also try: for each pair (v in not_in_set, u in in_set), if v conflicts only with u, 
        # swapping might help. But this is O(n^2), so only if time permits.
        if time.time() < polish_deadline - 1:
            in_set_indices = np.where(in_set)[0]
            not_in_set_indices = np.where(~in_set)[0]
            for v in not_in_set_indices:
                if time.time() > polish_deadline:
                    break
                k = count_conflicts[v]
                if k == 1:
                    # Find which in_set vertex v conflicts with
                    conflicting = np.where(conflicts[v] & in_set)[0]
                    if len(conflicting) == 1:
                        u = conflicting[0]
                        # Check if removing u and adding v enables more additions
                        # This is complex; skip for now
                        pass
    
    if in_set.sum() > best_size:
        best_size = in_set.sum()
        best_in_set = in_set
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
