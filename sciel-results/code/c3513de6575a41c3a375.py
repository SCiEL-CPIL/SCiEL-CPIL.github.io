"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time
from scipy.optimize import minimize

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
    T = np.zeros((Nm, Mp, Pn))
    
    idx_map = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                idx_map.append((u, v, w))
                
    # Initial best solution: Trivial Rank
    best_rank = R_trivial
    U_best = np.zeros((Nm, R_trivial))
    V_best = np.zeros((Mp, R_trivial))
    W_best = np.zeros((Pn, R_trivial))
    
    for r, (u, v, w) in enumerate(idx_map):
        U_best[u, r] = 1.0
        V_best[v, r] = 1.0
        W_best[w, r] = 1.0
        
    def _check_residual(U, V, W, tol=1e-9):
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T - T_approx)
        
    # Objective function for SciPy Optimizer
    # We flatten U, V, W into a single vector x
    # x = [U.flatten(), V.flatten(), W.flatten()]
    
    def objective(x):
        U = x[:Nm*r].reshape((Nm, r))
        V = x[Nm*r:Nm*r+Mp*r].reshape((Mp, r))
        W = x[Nm*r+Mp*r:].reshape((Pn, r))
        
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = T - T_approx
        return np.sum(diff**2)
        
    def gradient(x):
        U = x[:Nm*r].reshape((Nm, r))
        V = x[Nm*r:Nm*r+Mp*r].reshape((Mp, r))
        W = x[Nm*r+Mp*r:].reshape((Pn, r))
        
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = T - T_approx # T - T_approx
        
        # Gradient of 0.5 * ||T - T_approx||^2 w.r.t U
        # d/dU 0.5 * sum (T - UVW)^2
        # = - sum_{v,w} (T - T_approx) * V[v,r] * W[w,r]
        
        grad_U = -np.einsum('uvw,vr,wr->ur', diff, V, W)
        grad_V = -np.einsum('uvw,ur,wr->vr', diff, U, W)
        grad_W = -np.einsum('uvw,ur,vr->wr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Search for lower ranks
    # We try ranks from R_trivial - 1 down to a lower bound
    # Lower bound is max(Nm, Mp, Pn) = 28
    # We will try a few specific ranks and also a range
    
    min_rank_to_try = 28
    
    # To save time, we don't try every single rank from 83 down to 28.
    # We try a logarithmic scale or specific steps.
    # However, the jump from 84 to something lower is the main gain.
    # Let's try ranks: 83, 80, 70, 60, 50, 40, 35, 30, 28
    
    candidate_ranks = [83, 82, 80, 75, 70, 60, 50, 45, 40, 35, 30, 28]
    
    for r in candidate_ranks:
        if r >= best_rank:
            continue
            
        if time.time() > _LIMIT - 30.0:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(3):
            if time.time() > _LIMIT - 15.0:
                break
                
            seed = 42 + r * 100 + restart
            rs = np.random.RandomState(seed)
            
            # Initialize with small random values
            U = rs.standard_normal((Nm, r)) * 0.1
            V = rs.standard_normal((Mp, r)) * 0.1
            W = rs.standard_normal((Pn, r)) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
                
                if res.success or res.fun < 1e-8:
                    # Extract final U, V, W
                    U_final = res.x[:Nm*r].reshape((Nm, r))
                    V_final = res.x[Nm*r:Nm*r+Mp*r].reshape((Mp, r))
                    W_final = res.x[Nm*r+Mp*r:].reshape((Pn, r))
                    
                    # Verify exactness
                    if _check_residual(U_final, V_final, W_final) < 1e-7:
                        best_rank = r
                        best_U = U_final
                        best_V = V_final
                        best_W = W_final
                        found = True
                        break # Found for this rank, try lower
                        
            except Exception as e:
                pass
                
        if found:
            continue # Try next lower rank
        else:
            # If we fail at a high rank, lower ranks are likely harder.
            # But we have a sparse list of candidate ranks, so we just continue.
            pass

    return {"U": _pack(best_U), "V": _pack(best_V), "W": _pack(best_W)}
