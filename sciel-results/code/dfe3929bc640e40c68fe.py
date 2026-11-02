import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Candidates
    # Weight 1: (norm^2 = 4)
    v1 = []
    for pos in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            v1.append(v)
            
    # Weight 2: (norm^2 = 2)
    v2 = []
    for pos in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[pos[0]] = signs[0]
            v[pos[1]] = signs[1]
            v2.append(v)

    # Weight 4: (norm^2 = 4)
    v4 = []
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(pos, signs):
                v[i] = s
            v4.append(v)
            
    candidates = np.array(v1 + v2 + v4, dtype=np.int64)
    N = len(candidates)
    
    # 2. Compute Conflict Matrix
    # dist^2 >= max_norm^2
    # norms_sq = sum(candidates**2, axis=1)
    # dots = candidates @ candidates.T
    # conflict if dist^2 < max(norm_i^2, norm_j^2)
    # dist^2 = n_i^2 + n_j^2 - 2 dot
    # n_i^2 + n_j^2 - 2 dot < max(n_i^2, n_j^2)
    # 2 dot > n_i^2 + n_j^2 - max(n_i^2, n_j^2)
    # 2 dot > min(n_i^2, n_j^2)
    
    norms_sq = np.sum(candidates**2, axis=1)
    dots = candidates @ candidates.T
    
    # min_norm_sq for each pair is min(n_i, n_j)
    # We can compute min_norm_sq matrix
    # min_ns = np.minimum(norms_sq[:, None], norms_sq[None, :])
    # conflicts = dots * 2 > min_ns
    
    # Optimization: 
    # For n_i=2, n_j=2: 2 dot > 2 => dot > 1
    # For n_i=2, n_j=4: 2 dot > 2 => dot > 1
    # For n_i=4, n_j=4: 2 dot > 4 => dot > 2
    
    # Since dots are integers:
    # dot > 1 means dot >= 2
    # dot > 2 means dot >= 3
    
    # Let's just use the general formula to be safe
    min_ns = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflicts = (dots * 2) > min_ns
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    # neighbors[i] is a boolean mask or list of indices
    # Using boolean masks for fast operations in local search
    # But N=5500, so N*N booleans is 30MB, which is fine.
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_init():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # degrees[i] = number of neighbors of i that are alive
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            if time.time() > deadline - 5: break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            # Pick min degree
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            # Remove chosen and its neighbors from alive
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            # Update degrees
            # degrees[j] -= number of neighbors of j that were removed
            # This is expensive. 
            # Alternative: just recompute degrees for remaining? 
            # Or maintain a list of alive and check conflicts on the fly.
            
            # For speed, let's just do a simpler greedy:
            # Sort by degree, pick first, remove neighbors.
            # But degrees change.
            
            # Let's use a simpler local search on the result.
        return in_set

    def local_search(in_set_init):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        # count_conflicts[v] = number of neighbors of v in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[conflicts[i]] += 1 # vectorized addition
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Find best move: add v, remove neighbors
            # Gain = 1 - count_conflicts[v]
            # We want max gain > 0
            # If gain > 0, it's a strict improvement.
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            gains = 1 - count_conflicts[not_in_set]
            max_gain = np.max(gains)
            
            if max_gain > 0:
                # Pick one with max gain
                best_cands = not_in_set[gains == max_gain]
                v = best_cands[rng.integers(len(best_cands))]
                
                # Apply move
                to_remove = np.where(conflicts[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    # Update count_conflicts for neighbors of removed
                    for u in to_remove:
                        count_conflicts[conflicts[u]] -= 1
                
                in_set[v] = True
                current_size += max_gain
                count_conflicts[conflicts[v]] += 1
                improved = True
            else:
                # No strict improvement. Try random swap or accept neutral?
                # For now, break and rely on restarts.
                break
                
        return in_set, current_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        # Randomized Greedy Init
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            # Weighted random pick based on inverse degree?
            # Just min degree is robust.
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            # Update degrees efficiently?
            # degrees[j] for j in alive decreases by number of neighbors in to_remove
            # This is O(N) per step. Total O(N^2). 5500^2 = 30M. OK.
            if len(alive_indices) > 0:
                # Only update degrees for those still alive
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # Local Polish
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            polished, pol_size = local_search(in_set)
            if pol_size > best_size:
                best_size = pol_size
                best_in_set = polished.copy()

    final_indices = np.where(best_in_set)[0]
    return candidates[final_indices]
