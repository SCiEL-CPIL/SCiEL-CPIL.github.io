import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-flow_sums, kind="stable")
    loc_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Calculate initial cost
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    best_p = p.copy()
    best_cost = cost
    
    # Initialize S matrix: S[m, l] = sum_j A[m, j] * B[l, p[j]]
    # S = A @ B[:, p].T
    B_p = b[:, p]
    S = a @ B_p.T
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Tabu Search Parameters
    tabu_length = 50
    tabu_list = {}  # key: (i, k) with i < k, value: iteration count when it expires
    current_iter = 0
    
    # Precompute B columns for fast access in update?
    # B[:, v] - B[:, u] is needed.
    
    while time.time() - start_time < time_limit:
        # Find best move
        best_delta = 0
        best_i = -1
        best_k = -1
        best_is_tabu = False
        
        # Iterate over all pairs i, k
        for i in range(n):
            u = p[i]
            # S[i, u] is sum_j A[i, j] B[u, p[j]]
            # We want to minimize delta(i, k)
            # Delta(i, k) = (S[i, v] - S[i, u]) - (S[k, v] - S[k, u]) + corrections
            # v = p[k]
            
            # Vectorize over k?
            # For fixed i, u=p[i]
            # Term 1: S[i, p[k]] - S[i, u]
            # Term 2: S[k, p[k]] - S[k, u]
            
            # Let's just loop k for simplicity first, then optimize if needed.
            # With N=150, N^2=22500 iterations per pass.
            # Python loop might be slow. Let's try to vectorize the search for k for a given i.
            
            # k cannot be i
            # v = p[k]
            
            # Main part: (S[i, v] - S[i, u]) - (S[k, v] - S[k, u])
            # S[i, v] is S[i, p[k]]
            # S[k, v] is S[k, p[k]]
            # S[k, u] is S[k, u]
            
            # Let's compute arrays for all k != i
            # This is tricky to vectorize perfectly because p[k] changes.
            # But we can precompute S[i, p] which is the row S[i, :] indexed by p?
            # No, S[i, l] is for location l.
            # S[i, p[k]] is the value in row i, column p[k].
            
            # Let's stick to the loop for now, it's N^2.
            for k in range(n):
                if i == k:
                    continue
                v = p[k]
                
                # Check tabu
                ik = (i, k) if i < k else (k, i)
                is_tabu = ik in tabu_list and tabu_list[ik] > current_iter
                
                # Calculate Delta
                # Main part
                main_delta = (S[i, v] - S[i, u]) - (S[k, v] - S[k, u])
                
                # Corrections
                # j=i term: (a[i,i]-a[k,i]) * (b[v, u] - b[u, u])
                # j=k term: (a[i,k]-a[k,k]) * (b[v, v] - b[u, v])
                # Cross/Self:
                # a[i, k] * (b[v, u] - b[u, v])
                # a[k, i] * (b[u, v] - b[v, u])
                # a[i, i] * (b[v, v] - b[u, u])
                # a[k, k] * (b[u, u] - b[v, v])
                
                delta = main_delta
                delta -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
                delta -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
                delta += (a[i, k] * (b[v, u] - b[u, v]) + 
                         a[k, i] * (b[u, v] - b[v, u]) + 
                         a[i, i] * (b[v, v] - b[u, u]) + 
                         a[k, k] * (b[u, u] - b[v, v]))
                
                # Aspiration
                if is_tabu:
                    if cost + delta < best_cost:
                        is_tabu = False # Aspiration
                    else:
                        continue # Skip tabu move
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_k = k
                    best_is_tabu = is_tabu # Not strictly needed if we handle aspiration above
        
        # If no move found (shouldn't happen unless all are tabu and no aspiration)
        if best_i == -1:
            # Force a random move or restart?
            # In TS, we should always have a non-tabu move if tabu_length < N^2/2
            # If tabu list is full, we might need to clear it or pick the least tabu one.
            # For now, if best_i is -1, break or pick random.
            i = random.randint(0, n - 1)
            k = random.randint(0, n - 1)
            if i == k: continue
            # Calculate delta for this random move
            u = p[i]
            v = p[k]
            main_delta = (S[i, v] - S[i, u]) - (S[k, v] - S[k, u])
            delta = main_delta
            delta -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
            delta -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
            delta += (a[i, k] * (b[v, u] - b[u, v]) + 
                     a[k, i] * (b[u, v] - b[v, u]) + 
                     a[i, i] * (b[v, v] - b[u, u]) + 
                     a[k, k] * (b[u, u] - b[v, v]))
            best_i = i
            best_k = k
            best_delta = delta

        # Apply move
        i, k = best_i, best_k
        u = p[i]
        v = p[k]
        
        p[i], p[k] = v, u
        cost += best_delta
        
        # Update S
        # S += outer(A[:, i] - A[:, k], B[:, v] - B[:, u])
        diff_a = a[:, i] - a[:, k]
        diff_b = b[:, v] - b[:, u]
        S += np.outer(diff_a, diff_b)
        
        # Update Tabu List
        ik = (i, k) if i < k else (k, i)
        tabu_list[ik] = current_iter + tabu_length
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        current_iter += 1
        
        # Optional: Diversification if stuck?
        # Standard TS doesn't need it if tabu length is appropriate.
        
    return best_p.tolist()
