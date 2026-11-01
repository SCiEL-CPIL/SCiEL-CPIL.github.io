"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate W4 vectors (weight 4, entries in {-1, 1}, norm^2=4)
    v4_list = []
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            v4_list.append(v)
            
    V = np.array(v4_list, dtype=np.int64)
    N = len(V)
    
    # 2. Compute Conflict Matrix
    # Conflict if dot product > 2
    dots = V @ V.T
    conflicts = (dots > 2)
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists and counts
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # 3. Greedy Min-Degree Initialization
    def greedy_init():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degs = neighbor_counts.copy()
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = np.min(degs[alive_idx])
            cands = alive_idx[degs[alive_idx] == min_deg]
            v = cands[np.random.randint(len(cands))]
            
            in_set[v] = True
            nbrs = neighbors[v]
            to_remove = np.unique(np.concatenate(([v], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degs[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    # 4. Local Search (Simulated Annealing)
    def local_search(in_set_init, time_limit):
        in_set = in_set_init.copy()
        size = int(in_set.sum())
        
        # cc[v] = number of neighbors of v that are in in_set
        cc = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                cc[nbrs] += 1
        
        # S = sum of cc[v] for v in in_set
        S = int(cc[in_set].sum())
        
        T = 1.0
        cooling = 0.999
        lambda_pen = 5.0
        rng = np.random.default_rng(42)
        
        best_in_set = in_set.copy()
        best_size = size
        best_S = S
        
        end_time = start_time + time_limit
        
        while time.time() < end_time and T > 0.01:
            v = rng.integers(N)
            
            if in_set[v]:
                # Remove v
                # Delta E = 1 - 2 * lambda * cc[v]
                dE = 1.0 - 2.0 * lambda_pen * cc[v]
                
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    in_set[v] = False
                    size -= 1
                    S -= 2 * cc[v]
                    
                    nbrs = neighbors[v]
                    if len(nbrs) > 0:
                        cc[nbrs] -= 1
                    # cc[v] is no longer relevant for S, but we keep it for potential re-addition
                    # Actually, cc[v] should reflect the current state of neighbors.
                    # Since v is removed, its neighbors' cc decreased.
                    # cc[v] itself doesn't change (it counts neighbors in IS).
                    # But wait, if we remove v, the neighbors of v that are in IS lose a conflict.
                    # So cc[u] for u in nbrs(v) decreases.
                    # cc[v] remains the count of neighbors in IS.
                    # So cc[v] is correct.
                    
            else:
                # Add v
                # Delta E = -1 + 2 * lambda * cc[v]
                dE = -1.0 + 2.0 * lambda_pen * cc[v]
                
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    in_set[v] = True
                    size += 1
                    S += 2 * cc[v]
                    
                    nbrs = neighbors[v]
                    if len(nbrs) > 0:
                        cc[nbrs] += 1
                    # cc[v] remains the count of neighbors in IS.
                    # It was calculated before adding v.
                    # After adding v, the neighbors' cc increases.
                    # cc[v] is still the number of neighbors in IS.
                    # So it is correct.
            
            # Track best valid solution
            if S == 0 and size > best_size:
                best_size = size
                best_in_set = in_set.copy()
                best_S = 0
                
            T *= cooling
            
        # Polish: Repair and Fill
        # If S > 0, remove vertices with high conflict count until S == 0
        while S > 0:
            # Find v in in_set with max cc[v]
            in_set_idx = np.where(in_set)[0]
            if len(in_set_idx) == 0:
                break
            max_cc_idx = in_set_idx[np.argmax(cc[in_set_idx])]
            if cc[max_cc_idx] == 0:
                break # Should not happen if S > 0
            in_set[max_cc_idx] = False
            size -= 1
            S -= 2 * cc[max_cc_idx]
            nbrs = neighbors[max_cc_idx]
            if len(nbrs) > 0:
                cc[nbrs] -= 1
                
        # Greedy Fill: Add any vertex with cc[v] == 0
        candidates = np.where((cc == 0) & (~in_set))[0]
        for v in candidates:
            if cc[v] == 0 and not in_set[v]:
                in_set[v] = True
                size += 1
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    cc[nbrs] += 1
                # S remains 0 because we only add non-conflicting vertices
                # Wait, adding v increases cc of neighbors.
                # But v itself has cc[v]=0, so S doesn't change?
                # S = sum(cc[i] for i in IS).
                # We add v. cc[v] was 0. So S increases by 0.
                # But neighbors' cc increases.
                # Does that affect S?
                # S = sum_{i in IS} cc[i].
                # The new term is cc[v] = 0.
                # The old terms cc[u] for u in nbrs(v) increase by 1.
                # So S increases by len(nbrs(v) in IS).
                # This means S might become > 0!
                # So simple greedy fill on cc==0 is not safe if we track S strictly.
                # However, if cc[v]==0, v has no conflicts with IS.
                # So adding v does not create new conflicts for v.
                # But it increases the conflict count of its neighbors.
                # If a neighbor u was in IS, its cc[u] increases.
                # So S increases.
                # This means the set is no longer "valid" in terms of S=0?
                # No, S=0 means sum of conflicts is 0.
                # If we add v with cc[v]=0, v has 0 conflicts.
                # But u's conflict count increases.
                # This means u now has MORE conflicts?
                # No, cc[u] is the number of neighbors in IS.
                # If we add v, and v is a neighbor of u, then u gains a neighbor in IS.
                # So u now has a conflict?
                # Conflict definition: edge exists if dot > 2.
                # If (u,v) is an edge, they conflict.
                # If cc[v]==0, it means v has NO neighbors in IS.
                # So v is not connected to any u in IS.
                # So adding v does NOT increase cc[u] for any u in IS.
                # So S remains 0.
                # Correct.
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        return best_in_set

    # Main Loop
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Init
        in_set = greedy_init()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        # Allocate time based on restart number?
        # Just use a fixed chunk or remaining time.
        # Let's use a fixed 1s per SA run, or scale with remaining.
        # To be safe, use min(1.0, remaining_time/2)
        sa_time = min(1.5, remaining_time / 2)
        
        in_set = local_search(in_set, sa_time)
        
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # 3. Perturb Best
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                # Remove 2-5 random nodes
                k_remove = np.random.randint(2, 6)
                to_remove = np.random.choice(indices, size=k_remove, replace=False)
                perturbed[to_remove] = False
                perturbed = local_search(perturbed, sa_time)
                current_size = int(perturbed.sum())
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = perturbed.copy()

    # Final Result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = V[final_v4_indices]
    
    # Add W1 vectors
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
