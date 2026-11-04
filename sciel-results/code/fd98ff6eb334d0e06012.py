"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_rank_als(r, n, m, p, T_true, max_iters=2000, seed=42, time_budget=30):
    """Try to find rank-r decomposition using ALS with half-integer rounding."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    for init_seed in range(3):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + init_seed)
        
        # Initialize with random values
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Check if we're close enough
            err = np.max(np.abs(R))
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
                
                # If rounding didn't work, continue with original
                if err < 1e-8:
                    # Keep original values if they work exactly
                    best_solution = (U, V, W)
                    best_err = err
            
            # ALS updates
            # Update U: for each a, solve (V*W^T) U[a,:]^T = T_true[a,:,:] weighted
            # Simpler: gradient-based update
            eta = 0.05
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Periodically check and try rounding
            if it % 50 == 0 and err < 0.1:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound is max(nm, mp, pn) = 32
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound up to best_rank-1
    # Limit total search time to 100 seconds (leaving 20s safety margin)
    search_start = time.time()
    max_search_time = 100.0
    
    # Try a subset of ranks to stay within time budget
    # Focus on ranks that are "structured" (multiples of small numbers)
    candidate_ranks = []
    for r in range(lower_bound, best_rank):
        candidate_ranks.append(r)
    
    # Sort by likelihood of success (try lower ranks first)
    for r in candidate_ranks:
        if time.time() - search_start > max_search_time:
            break
        
        if r >= best_rank:
            break
        
        # Try to find a rank-r decomposition
        result = _try_rank_als(r, n, m, p, T_true, 
                               max_iters=1000, seed=42, 
                               time_budget=15)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = r
                # If we found a good solution, we can stop early
                # or continue to try even lower ranks
                if r <= lower_bound + 10:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
