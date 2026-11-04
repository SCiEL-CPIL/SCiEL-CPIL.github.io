"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 130.0  # 20s safety margin before 150s kill
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    
    # Fast residual computation
    def residual_fast(U, V, W):
        r = U.shape[1]
        # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum for efficiency
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((T_decomp - T) ** 2)
    
    # Alternating Least Squares with proper linear solves
    def als_optimize(U_init, V_init, W_init, target_rank, max_outer=20, max_inner=5):
        r = target_rank
        U = U_init[:, :r].copy()
        V = V_init[:, :r].copy()
        W = W_init[:, :r].copy()
        
        nm, mp, pn = n*m, m*p, p*n
        
        for outer in range(max_outer):
            if time.time() > deadline:
                break
            
            # Update U: for each a, solve M_a @ U[a,:] = T[a].reshape(-1)
            # where M_a[(b,c), r'] = V[b,r'] * W[c,r']
            VW = np.einsum('br,cr->bcr', V, W, optimize=True).reshape(mp*pn, r)
            for a in range(nm):
                t_a = T[a].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(VW, t_a, rcond=None)[0]
                except:
                    pass
            
            # Update V: for each b, solve M_b @ V[b,:] = T[:,b,:].reshape(-1)
            # where M_b[(a,c), r'] = U[a,r'] * W[c,r']
            UW = np.einsum('ar,cr->acr', U, W, optimize=True).reshape(nm*pn, r)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(UW, t_b, rcond=None)[0]
                except:
                    pass
            
            # Update W: for each c, solve M_c @ W[c,:] = T[:,:,c].reshape(-1)
            # where M_c[(a,b), r'] = U[a,r'] * V[b,r']
            UV = np.einsum('ar,br->abr', U, V, optimize=True).reshape(nm*mp, r)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(UV, t_c, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            res = residual_fast(U, V, W)
            if res < 1e-10:
                break
        
        return U, V, W, residual_fast(U, V, W)
    
    # Try to find a decomposition with rank < 120
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_schoolbook
    best_res = 0.0
    
    # Try various target ranks
    for target_rank in [100, 95, 90, 85, 80, 75, 70, 65, 60]:
        if time.time() > deadline - 20:
            break
        
        U_t, V_t, W_t, res = als_optimize(U0, V0, W0, target_rank, max_outer=15)
        
        if res < 1e-6:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            best_res = res
            break
    
    # Round to half-integers and verify
    def round_to_half_integer(x):
        return np.round(x * 2) / 2
    
    U_rounded = round_to_half_integer(best_U)
    V_rounded = round_to_half_integer(best_V)
    W_rounded = round_to_half_integer(best_W)
    
    # Check if rounded version is still valid
    res_rounded = residual_fast(U_rounded, V_rounded, W_rounded)
    
    if res_rounded < 1e-6:
        U_final, V_final, W_final = U_rounded, V_rounded, W_rounded
    else:
        # Try to fix with a few more ALS iterations on the rounded values
        U_fix, V_fix, W_fix, res_fix = als_optimize(U_rounded, V_rounded, W_rounded, best_rank, max_outer=5)
        res_fix_rounded = residual_fast(round_to_half_integer(U_fix), round_to_half_integer(V_fix), round_to_half_integer(W_fix))
        if res_fix_rounded < res_rounded:
            U_final = round_to_half_integer(U_fix)
            V_final = round_to_half_integer(V_fix)
            W_final = round_to_half_integer(W_fix)
        else:
            U_final, V_final, W_final = best_U, best_V, best_W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
