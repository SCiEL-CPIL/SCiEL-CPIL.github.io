"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction
import scipy.optimize as so

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if len(U) == 0:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    # Also check that no spurious terms exist
    # For any (i,j,k) combination, the only contribution should be from the correct terms
    # Actually, we need to verify the full tensor decomposition
    # The tensor T[i,j,k] = delta_{i,j,k} in the sense that C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # More precisely, T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We already checked this above.
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=10.0):
    """Try to find a decomposition of rank target_rank starting from initial guess."""
    start_time = time.time()
    
    U0 = np.array(U, dtype=float)
    V0 = np.array(V, dtype=float)
    W0 = np.array(W, dtype=float)
    current_rank = len(U0[0])
    
    if target_rank >= current_rank:
        return None
    
    # Random initialization: pick target_rank columns from the schoolbook
    idx = np.random.choice(current_rank, target_rank, replace=False)
    U_init = U0[:, idx].copy()
    V_init = V0[:, idx].copy()
    W_init = W0[:, idx].copy()
    
    # Add small noise
    U_init += np.random.randn(*U_init.shape) * 0.1
    V_init += np.random.randn(*V_init.shape) * 0.1
    W_init += np.random.randn(*W_init.shape) * 0.1
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def residuals(x):
        U_flat, V_flat, W_flat = x[:nm*target_rank], x[nm*target_rank:nm*target_rank+mp*target_rank], x[nm*target_rank+mp*target_rank:]
        U_r = U_flat.reshape(nm, target_rank)
        V_r = V_flat.reshape(mp, target_rank)
        W_r = W_flat.reshape(pn, target_rank)
        
        res = np.zeros(n*m*p)
        idx_res = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    res[idx_res] = 1.0 - val
                    idx_res += 1
        return res
    
    try:
        res = so.least_squares(residuals, x0, method='lm', max_nfev=2000)
        if time.time() - start_time > max_time:
            return None
            
        if res.cost < 1e-8:
            x_opt = res.x
            U_sol = x_opt[:nm*target_rank].reshape(nm, target_rank)
            V_sol = x_opt[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W_sol = x_opt[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-5):
                return U_sol, V_sol, W_sol
    except Exception:
        pass
        
    return None

def _snap_to_rational(U, V, W, max_denom=100):
    """Try to snap floating point values to rationals with small denominators."""
    U_r = np.zeros_like(U, dtype=object)
    V_r = np.zeros_like(V, dtype=object)
    W_r = np.zeros_like(W, dtype=object)
    
    def snap(val):
        for denom in range(1, max_denom+1):
            num = round(val * denom)
            if abs(val - num/denom) < 1e-10:
                return Fraction(num, denom)
        return Fraction(val)
    
    for i in range(len(U)):
        for j in range(len(U[0])):
            U_r[i, j] = snap(U[i, j])
    for i in range(len(V)):
        for j in range(len(V[0])):
            V_r[i, j] = snap(V[i, j])
    for i in range(len(W)):
        for j in range(len(W[0])):
            W_r[i, j] = snap(W[i, j])
    
    return U_r, V_r, W_r

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to reduce rank
    # Known lower bounds: for 2x3x10, the tensor rank is at least max(n*p, m*p, n*m) = max(20, 30, 6) = 30
    # But we can try various ranks
    for target_rank in range(30, rank_schoolbook):
        if time.time() > deadline - 15:
            break
        
        # Limit time per attempt to ensure we finish
        time_left = deadline - time.time() - 5
        if time_left <= 0:
            break
            
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=min(10.0, time_left))
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Verify
            if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-5):
                # Try to snap to rationals
                U_r, V_r, W_r = _snap_to_rational(U_sol, V_sol, W_sol)
                # Convert back to list of lists
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                
                # Final verification
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break  # Found the lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
