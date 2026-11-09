import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix
    # D[i, j] = squared distance between point i and point j
    # Use broadcasting to compute
    # points is (M, 2)
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # (M, M, 2)
    # D = np.sum(diff**2, axis=2) # (M, M)
    # This might take a moment and ~200MB RAM.
    
    try:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        D = np.sum(diff**2, axis=2).astype(np.int32)
    except MemoryError:
        # Fallback to on-the-fly if memory fails, though 67MB is safe
        D = None

    def can_add(p, S_list):
        if not S_list:
            return True
            
        s_indices = np.array(S_list)
        
        # 1. Distances from p to all in S must be distinct
        if D is not None:
            d_p_S = D[p, s_indices]
            # Check for duplicates in d_p_S
            # Since d_p_S are integers, we can sort and check neighbors
            # Or use np.unique
            # np.unique is O(k log k). For k ~ 100, this is fine.
            if len(np.unique(d_p_S)) != len(s_indices):
                return False
            
            # 2. For each s in S, d(p, s) must not equal d(s, t) for any t in S, t != s
            # d_S_S is the submatrix of D for S
            d_S_S = D[np.ix_(s_indices, s_indices)]
            
            # We need to check if d_p_S[k] is in d_S_S[k, :] (excluding k itself? No, d_S_S[k,k]=0, d_p_S[k]>0)
            # So just check if d_p_S[k] is in d_S_S[k, :].
            # Vectorized approach:
            # Create a boolean matrix where B[k, j] = (d_p_S[k] == d_S_S[k, j])
            # If any row has any True, it's invalid.
            # Note: d_p_S[k] is the distance from p to s_k.
            # d_S_S[k, j] is the distance from s_k to s_j.
            # If d_p_S[k] == d_S_S[k, j], then triangle (p, s_k, s_j) is isosceles with legs (p, s_k) and (s_k, s_j).
            
            # Let's construct the comparison efficiently.
            # d_p_S is shape (K,)
            # d_S_S is shape (K, K)
            # We want to compare d_p_S[i] with d_S_S[i, :] for each i.
            # This is equivalent to checking if d_p_S is in the set of distances from each s_i to all other s_j.
            
            # We can do:
            # matches = (d_S_S == d_p_S[:, np.newaxis])
            # if matches.any(): return False
            
            matches = (d_S_S == d_p_S[:, np.newaxis])
            if matches.any():
                return False
        else:
            # Fallback
            def get_dist(p_idx, s_idx_list):
                diff = points[p_idx] - points[s_idx_list]
                return np.sum(diff**2, axis=1)

            d_p_S = get_dist(p, s_indices)
            if len(np.unique(d_p_S)) != len(s_indices):
                return False
            
            for k in range(len(s_indices)):
                d_val = d_p_S[k]
                # Check against all other distances from s_k
                dists_from_k = get_dist(s_indices[k], s_indices)
                if d_val in dists_from_k:
                    return False
        
        return True

    best_S = []
    best_size = 0
    
    # Initial greedy
    S = []
    candidates = list(range(M))
    random.shuffle(candidates)
    for p in candidates:
        if can_add(p, S):
            S.append(p)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()

    while time.time() < deadline - 5:
        # Perturbation: Take a random subset of best_S
        # Size of subset: between 0 and len(best_S)
        # Or just keep all and try to improve?
        # Let's try a mix:
        
        # Strategy A: Full Restart with Random Order
        # Strategy B: Perturb Best
        
        if random.random() < 0.5 and best_size > 0:
            # Perturb Best
            # Remove a few random points
            S = best_S.copy()
            remove_count = random.randint(1, min(5, len(S)))
            to_remove = random.sample(S, remove_count)
            for r in to_remove:
                S.remove(r)
            
            # Try to add points from the rest of the grid
            # Prioritize points that were removed? Or random?
            # Let's shuffle all candidates and try to add
            candidates = [p for p in range(M) if p not in S]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(p, S):
                    S.append(p)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
        else:
            # Full Restart
            S = []
            candidates = list(range(M))
            random.shuffle(candidates)
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(p, S):
                    S.append(p)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()

    return [tuple(points[i]) for i in best_S]
