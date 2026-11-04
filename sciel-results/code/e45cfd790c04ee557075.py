"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 140.0          # hard safety margin before the 150 s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _pack(arr):
    return [[float(x) for x in row] for row in np.asarray(arr)]


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # Trivial rank is n*m*p
    R_trivial = n * m * p
    
    # Build the multiplication tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Otherwise 0
    T = np.zeros((Nm, Mp, Pn))
    
    # Map (i, j, k) to index r in trivial decomposition
    # and populate T
    idx_map = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                idx_map.append((u, v, w))
                
    # Initialize best solution with trivial rank
    U_best = np.zeros((Nm, R_trivial))
    V_best = np.zeros((Mp, R_trivial))
    W_best = np.zeros((Pn, R_trivial))
    
    for r, (u, v, w) in enumerate(idx_map):
        U_best[u, r] = 1.0
        V_best[v, r] = 1.0
        W_best[w, r] = 1.0
        
    best_rank = R_trivial
    best_U = U_best
    best_V = V_best
    best_W = W_best
    
    # Function to check exactness
    def _check_residual(U, V, W, tol=1e-9):
        # Reconstruct T_approx
        # T_approx[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T - T_approx)
        
    # Try to find a lower rank
    # Search ranks from R_trivial - 1 down to some lower bound
    # Lower bound for matmul rank is max(Nm, Mp, Pn) = 28
    # But we can't expect to find it easily.
    # Let's try a few ranks below trivial.
    
    min_rank_to_try = max(28, R_trivial - 5) # Only try a few steps down to save time
    
    for r in range(R_trivial - 1, min_rank_to_try - 1, -1):
        if time.time() > _LIMIT - 20.0:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() > _LIMIT - 10.0:
                break
                
            seed = 42 + r * 100 + restart
            rs = np.random.RandomState(seed)
            
            # Initialize
            U = rs.standard_normal((Nm, r))
            V = rs.standard_normal((Mp, r))
            W = rs.standard_normal((Pn, r))
            
            # ALS iterations
            converged = False
            for it in range(50):
                # Update U
                for rr in range(r):
                    # U[:, rr] = argmin ||T - sum_r ...||_U
                    # T_contracted[u] = sum_{v,w} T[u,v,w] V[v,rr] W[w,rr]
                    # Since T is sparse, we can iterate over non-zeros
                    # But T is small, let's use dense ops
                    T_contracted = np.einsum('uvw,vr,wr->ur', T, V, W)
                    denom = np.sum(V[:, rr]**2) * np.sum(W[:, rr]**2)
                    if denom > 1e-12:
                        U[:, rr] = T_contracted[:, rr] / denom
                        
                # Update V
                for rr in range(r):
                    T_contracted = np.einsum('uvw,ur,wr->vr', T, U, W)
                    denom = np.sum(U[:, rr]**2) * np.sum(W[:, rr]**2)
                    if denom > 1e-12:
                        V[:, rr] = T_contracted[:, rr] / denom
                        
                # Update W
                for rr in range(r):
                    T_contracted = np.einsum('uvw,ur,vr->wr', T, U, V)
                    denom = np.sum(U[:, rr]**2) * np.sum(V[:, rr]**2)
                    if denom > 1e-12:
                        W[:, rr] = T_contracted[:, rr] / denom
                        
                if it > 0 and it % 10 == 0:
                    res = _check_residual(U, V, W)
                    if res < 1e-8:
                        converged = True
                        break
                        
            if converged:
                # Verify again
                if _check_residual(U, V, W) < 1e-8:
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break # Found for this rank, move to next lower rank
                    
        if found:
            continue # Try lower rank
        else:
            # If not found at this rank, maybe lower ranks are harder?
            # Usually if you can't find rank R, you might not find R-1 easily either.
            # But it's possible.
            # However, to save time, we might stop if we fail for a few ranks.
            # For now, just continue.
            pass

    return {"U": _pack(best_U), "V": _pack(best_V), "W": _pack(best_W)}
