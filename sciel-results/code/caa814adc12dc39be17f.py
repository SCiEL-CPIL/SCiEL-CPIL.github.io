import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120  # seconds
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Try to find a rank-44 decomposition using iterative refinement
    if time.time() - start_time < time_limit - 30:
        try:
            target_rank = 44
            rng = np.random.RandomState(42)
            
            # Build the target tensor
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1
            
            # Initialize with schoolbook and try to reduce
            # Use alternating minimization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with first target_rank terms of schoolbook
            for r in range(target_rank):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            
            # Alternating minimization
            for iteration in range(100):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Update W: for each c, solve min_W sum_{a,b} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # This is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For fixed c, this is a linear system in W[c,:]
                
                # Build system: for each (a,b), sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
                # Let M[(a,b),r] = U[a,r]*V[b,r], then M @ W[c,:] = T[:,:,c].flatten()
                
                M_W = np.zeros((n*m*m*p, target_rank))
                b_W = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for r in range(target_rank):
                            M_W[idx, r] = U[a, r] * V[b, r]
                        b_W[idx] = T[a, b, c] if False else 0
                        idx += 1
                
                # Actually, for each c, we need T[a,b,c]
                # Let's redo this properly
                
                # For updating W[c,:], we fix a,b and vary c
                # The equation is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For fixed c, this gives: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
                # So for each c, we have a system: M @ W[c,:] = t_c where M[(a,b),r]=U[a,r]*V[b,r] and t_c[(a,b)]=T[a,b,c]
                
                M_W = np.zeros((n*m*m*p, target_rank))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for r in range(target_rank):
                            M_W[idx, r] = U[a, r] * V[b, r]
                        idx += 1
                
                for c in range(p*n):
                    t_c = np.zeros(n*m*m*p)
                    idx = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            t_c[idx] = T[a, b, c]
                            idx += 1
                    W[c, :], res, rank, sv = np.linalg.lstsq(M_W, t_c, rcond=None)
                
                # Update U: for each a, solve min sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # For fixed a: sum_r U[a,r] * (sum_b V[b,r]*W[c,r]) = T[a,b,c]
                # Let's define for each (b,c): M_U[(b,c),r] = V[b,r]*W[c,r]
                # Then for each a: M_U @ U[a,:] = T[a,:,:].flatten()
                
                M_U = np.zeros((m*p*p*n, target_rank))
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        for r in range(target_rank):
                            M_U[idx, r] = V[b, r] * W[c, r]
                        idx += 1
                
                for a in range(n*m):
                    t_a = np.zeros(m*p*p*n)
                    idx = 0
                    for b in range(m*p):
                        for c in range(p*n):
                            t_a[idx] = T[a, b, c]
                            idx += 1
                    U[a, :], res, rank, sv = np.linalg.lstsq(M_U, t_a, rcond=None)
                
                # Update V: for each b, solve min sum_{a,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # For fixed b: sum_r V[b,r] * (sum_a U[a,r]*W[c,r]) = T[a,b,c]
                # Let's define for each (a,c): M_V[(a,c),r] = U[a,r]*W[c,r]
                # Then for each b: M_V @ V[b,:] = T[:,b,:].flatten()
                
                M_V = np.zeros((n*m*p*n, target_rank))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        for r in range(target_rank):
                            M_V[idx, r] = U[a, r] * W[c, r]
                        idx += 1
                
                for b in range(m*p):
                    t_b = np.zeros(n*m*p*n)
                    idx = 0
                    for a in range(n*m):
                        for c in range(p*n):
                            t_b[idx] = T[a, b, c]
                            idx += 1
                    V[b, :], res, rank, sv = np.linalg.lstsq(M_V, t_b, rcond=None)
                
                # Check error
                E = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    E += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.sum((T - E) ** 2)
                
                if err < 1e-10:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    break
        
        except Exception as e:
            pass
    
    if time.time() - start_time > time_limit:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
