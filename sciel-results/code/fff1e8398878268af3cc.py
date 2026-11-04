import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute all indices and targets
    # u in [0, nm), v in [0, mp), w in [0, pn)
    # Target T[u, v, w] is 1 if indices are consistent, 0 otherwise.
    # Consistency:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # So j = u % m, i = u // m
    # j' = v // p, k = v % p
    # k' = w % n, i' = w // n
    # Check: j == j' and k == k' and i == i'
    
    targets = np.zeros((nm, mp, pn))
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            j_prime = v // p
            k = v % p
            if j != j_prime:
                continue
            for w in range(pn):
                k_prime = w % n
                i_prime = w // n
                if k == k_prime and i == i_prime:
                    targets[u, v, w] = 1.0

    # Flatten targets for residual
    target_flat = targets.flatten()
    num_eq = len(target_flat)
    
    def residual(params, rank):
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        
        # Compute sum_r U[u,r] V[v,r] W[w,r] for all u,v,w
        # This is a tensor contraction.
        # We can do this efficiently using einsum or loops.
        # result[u,v] = sum_r U[u,r] V[v,r]
        # Then res[u,v,w] = sum_r (U[u,r] V[v,r]) W[w,r]
        # This is not quite right. The sum is over r for the triple product.
        # res[u,v,w] = dot(U[u,:], V[v,:] * W[w,:])
        
        # Vectorized approach:
        # U: (nm, rank)
        # V: (mp, rank)
        # W: (pn, rank)
        # We want T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        
        # Intermediate: M[u, v, r] = U[u, r] * V[v, r]
        # This is huge: nm * mp * rank.
        # Then T[u,v,w] = sum_r M[u,v,r] W[w,r]
        
        # Alternative:
        # For each r, T_r = outer(outer(U[:,r], V[:,r]), W[:,r])
        # Sum them.
        
        T = np.zeros((nm, mp, pn))
        for r in range(rank):
            # u, v, w
            # T[:, :, :] += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
            # This is efficient enough for small ranks.
            T += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
            
        return T.flatten() - target_flat

    # Try ranks from low to high
    # We know rank 24 works.
    # Try 23, 22, ...
    
    for rank in range(23, 14, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        # Estimate time per attempt. 
        # least_squares with 576 equations and 26*R variables.
        # Let's allow a few attempts.
        max_attempts = max(1, int(remaining / 5))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 10000 + attempt * 7 + 42)
            # Initialize with small random values
            scale = 1.0
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use 'trf' method which handles bounds and is robust
                # No bounds specified, so it's unconstrained
                result = least_squares(residual, x0, args=(rank,), method='trf', max_nfev=200)
                
                # Check cost
                if result.cost < 1e-10:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    # Verify max error
                    max_err = 0.0
                    for u in range(nm):
                        for v in range(mp):
                            for w in range(pn):
                                val = np.dot(U_sol[u, :], V_sol[v, :] * W_sol[w, :])
                                err = abs(val - targets[u, v, w])
                                if err > max_err:
                                    max_err = err
                                    
                    if max_err < 1e-8:
                        # Try to round to half-integers
                        def to_list(arr):
                            lst = []
                            for row in arr:
                                row_lst = []
                                for val in row:
                                    rounded = round(val * 2) / 2
                                    if abs(val - rounded) < 1e-6:
                                        row_lst.append(rounded)
                                    else:
                                        row_lst.append(float(val))
                                lst.append(row_lst)
                            return lst
                        
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue
    
    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
