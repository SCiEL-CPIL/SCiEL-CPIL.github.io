"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [3, 4, 7]

def _verify_exact(U, V, W, n, m, p):
    """Verify exactness with Fraction arithmetic."""
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m: return False
    if len(V) != m * p: return False
    if len(W) != p * n: return False
    
    for arr in [U, V, W]:
        if len(arr) == 0: return False
        if len(arr[0]) != rank: return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i*m+j]
                v_row = V[j*p+k]
                w_row = W[k*n+i]
                for c in range(rank):
                    s += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                if s != Fraction(1):
                    return False
    return True

def _residual(params, n, m, p, rank):
    """Compute residuals for the tensor decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    
    residuals = np.empty(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                residuals[idx] = np.dot(u_row, v_row * w_row) - 1.0
                idx += 1
    return residuals

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Fallback result
    fallback = _schoolbook(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 130.0 # Leave 20s for safety
    
    # Try ranks from schoolbook_rank - 1 downwards
    # We expect a solution to be close to schoolbook if it exists
    for rank in range(schoolbook_rank - 1, 69, -1):
        if time.time() > deadline:
            break
            
        # Dynamic number of restarts based on time remaining
        # Estimate 1s per restart
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        n_restarts = int(time_left / 1.5)
        n_restarts = max(5, min(n_restarts, 50)) # Cap restarts to avoid over-spending on one rank
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Use different seeds for each rank and restart
            seed_val = rank * 1000 + r
            np.random.seed(seed_val)
            
            # Initialize with small random values
            # Scale matters: if values are too large, gradients explode.
            # If too small, gradients vanish.
            # 0.1 to 1.0 is a good range.
            params_init = np.random.randn((nm + mp + pn) * rank) * 0.5
            
            try:
                # Use least_squares with 'trf' method (Trust Region Reflective)
                # loss='linear' is standard L2, which is appropriate here
                result = opt.least_squares(
                    _residual,
                    params_init,
                    args=(n, m, p, rank),
                    method='trf',
                    loss='linear',
                    f_scale=1.0,
                    max_nfev=500 # Limit iterations to ensure speed
                )
                
                if result.cost < 1e-8:
                    # Check if we can snap to half-integers
                    U_flat = result.x[:nm * rank]
                    V_flat = result.x[nm * rank: nm * rank + mp * rank]
                    W_flat = result.x[nm * rank + mp * rank:]
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_flat * 2) / 2
                    V_snap = np.round(V_flat * 2) / 2
                    W_snap = np.round(W_flat * 2) / 2
                    
                    U_list = U_snap.reshape(nm, rank).tolist()
                    V_list = V_snap.reshape(mp, rank).tolist()
                    W_list = W_snap.reshape(pn, rank).tolist()
                    
                    # Verify exactness
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        return {
                            "U": U_list,
                            "V": V_list,
                            "W": W_list
                        }
                        
            except Exception:
                pass

    return fallback

# EVOLVE-BLOCK-END
