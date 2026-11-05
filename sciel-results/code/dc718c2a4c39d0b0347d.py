import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    vals = np.arange(1, N + 1)
    
    # Build Conflict Graph
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        a = vals[i]
        b = vals
        sums = a + b
        prods = a * b
        mod = prods % sums
        A[i, :] = (mod == 0)
        
    # Initial Solution: All Odd Numbers
    S = np.zeros(N, dtype=bool)
    S[::2] = True 
    current_size = np.sum(S)
    best_S = S.copy()
    best_size = current_size
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute neighbors for faster lookup? A is already a boolean matrix.
    # A[v, :] is the neighbor list of v.
    
    while time.time() < deadline:
        # 1. Try to add any possible node
        if np.sum(S) > 0:
            has_conflict = A[:, S].any(axis=1)
        else:
            has_conflict = np.zeros(N, dtype=bool)
        
        candidates_add = ~S & ~has_conflict
        if np.any(candidates_add):
            idxs = np.where(candidates_add)[0]
            v = idxs[random.randint(0, len(idxs) - 1)]
            S[v] = True
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
            continue
        
        # 2. Try targeted swaps
        # For each u in S, find v not in S such that N_S(v) == {u}
        # This means v is blocked ONLY by u.
        # If we remove u, v can be added. Size stays same (remove u, add v).
        # But it might open up paths for future adds.
        
        # Efficient way:
        # For each v not in S, calculate count of neighbors in S.
        # neighbor_counts = A[:, S].sum(axis=1) # Shape (N,)
        # If neighbor_counts[v] == 1, find the neighbor u.
        
        if np.sum(S) > 0:
            neighbor_counts = A[:, S].sum(axis=1)
        else:
            neighbor_counts = np.zeros(N, dtype=int)
            
        # Candidates for "blocked by exactly 1"
        # v must not be in S
        # neighbor_counts[v] == 1
        potential_v = (~S) & (neighbor_counts == 1)
        
        if np.any(potential_v):
            # Pick a random v from potential_v
            v_idxs = np.where(potential_v)[0]
            v = v_idxs[random.randint(0, len(v_idxs) - 1)]
            
            # Find the unique neighbor u in S
            # A[v, S] is True for exactly one index.
            # Find that index.
            # We can use np.where(A[v, S])[0] to get relative indices in S.
            # Then map back to global index.
            s_idxs = np.where(S)[0]
            rel_idx = np.where(A[v, s_idxs])[0][0]
            u = s_idxs[rel_idx]
            
            # Perform swap: remove u, add v
            S[u] = False
            S[v] = True
            # current_size unchanged
            
            # Check if this swap allows an immediate add?
            # The loop will restart and check for adds.
            
            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T
        else:
            # No targeted swaps. Try random swap or cool down.
            # Random swap: remove random u, add random v that doesn't conflict with S\{u}
            s_idxs = np.where(S)[0]
            if len(s_idxs) == 0:
                break
                
            u = s_idxs[random.randint(0, len(s_idxs) - 1)]
            S[u] = False
            current_size -= 1
            
            # Find v to add
            if np.sum(S) > 0:
                has_conflict = A[:, S].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
                
            candidates = ~S & ~has_conflict
            if np.any(candidates):
                c_idxs = np.where(candidates)[0]
                v = c_idxs[random.randint(0, len(c_idxs) - 1)]
                S[v] = True
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Revert
                S[u] = True
                current_size += 1
                
            T *= cooling
            if T < min_T:
                T = min_T

    result = [i + 1 for i in range(N) if best_S[i]]
    if not result:
        return [1]
    return result

