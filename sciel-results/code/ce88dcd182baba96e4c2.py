"""
Optimized tensor decomposition for 3x4 * 4x4 matrix multiplication.
Uses Nelder-Mead optimization with restarts and rounding/polishing to find low-rank exact decompositions.
"""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

# EVOLVE-BLOCK-START

def _verify_einsum(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
        
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False

    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # The tensor is T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want T[i,j,k] = 1 for all i,j,k
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if all entries are close to 1
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)

def _polish_solution(U, V, W, n, m, p, max_iter=100):
    """
    Attempt to polish a near-exact solution to an exact one.
    We round to integers first, then do a few L-BFGS-B steps to fine-tune.
    """
    # Round to nearest integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    
    # Check if rounded solution is already good
    if _verify_einsum(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return U_int, V_int, W_int
    
    # If not, try to polish from the integer starting point
    # Flatten
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    x0 = np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()])
    
    def loss_func(x):
        U_flat = x[:nm*r]
        V_flat = x[nm*r:nm*r+mp*r]
        W_flat = x[nm*r+mp*r:]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.sum((recon - 1.0) ** 2)
        
    try:
        res = minimize(loss_func, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < 1e-9:
            x_best = res.x
            U_best = x_best[:nm*r].reshape(nm, r)
            V_best = x_best[nm*r:nm*r+mp*r].reshape(mp, r)
            W_best = x_best[nm*r+mp*r:].reshape(pn, r)
            
            # Final rounding check
            U_final = np.round(U_best)
            V_final = np.round(V_best)
            W_final = np.round(W_best)
            
            if _verify_einsum(U_final, V_final, W_final, n, m, p, tol=1e-9):
                return U_final, V_final, W_final
            
            # If rounded version isn't perfect but the float version is, return float?
            # The task asks for integer/half-integer entries.
            # If float is exact but not integer, it might be valid if half-integers are allowed?
            # "integer/half-integer entries"
            # Let's check if 2*x is integer.
            U_half = U_best * 2
            V_half = V_best * 2
            W_half = W_best * 2
            
            if np.allclose(U_half, np.round(U_half), atol=1e-9) and \
               np.allclose(V_half, np.round(V_half), atol=1e-9) and \
               np.allclose(W_half, np.round(W_half), atol=1e-9):
                return U_best, V_best, W_best

    except Exception:
        pass
        
    return None

def _search_rank(n, m, p, rank, time_budget, rng):
    """
    Try to find a decomposition of a specific rank using Nelder-Mead.
    """
    nm, mp, pn = n*m, m*p, p*n
    total_vars = (nm + mp + pn) * rank
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Number of restarts depends on time
    # Each Nelder-Mead run takes some time. Let's assume ~0.5s per run for small ranks.
    # We'll just run as many as we can.
    
    best_result = None
    best_loss = np.inf
    
    while time.time() < deadline - 1.0:
        # Random initialization
        # Scale: values should be around 1 or -1
        U_init = rng.uniform(-1, 1, (nm, rank))
        V_init = rng.uniform(-1, 1, (mp, rank))
        W_init = rng.uniform(-1, 1, (pn, rank))
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def loss_func(x):
            U_flat = x[:nm*rank]
            V_flat = x[nm*rank:nm*rank+mp*rank]
            W_flat = x[nm*rank+mp*rank:]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return np.sum((recon - 1.0) ** 2)
            
        try:
            res = minimize(loss_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
            
            if res.fun < best_loss:
                best_loss = res.fun
                x_best = res.x
                U_best = x_best[:nm*rank].reshape(nm, rank)
                V_best = x_best[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_best = x_best[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Try to polish
                polished = _polish_solution(U_best, V_best, W_best, n, m, p)
                if polished is not None:
                    U_p, V_p, W_p = polished
                    if _verify_einsum(U_p, V_p, W_p, n, m, p, tol=1e-9):
                        return U_p.tolist(), V_p.tolist(), W_p.tolist(), rank
                        
        except Exception:
            pass
            
    # If we didn't find an exact solution, return None
    return None

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Lower bound is 27. Upper bound (schoolbook) is 48.
    # We will search from low to high rank.
    # If we find a solution at rank R, we stop.
    
    # Use a fixed seed for reproducibility of the search process itself?
    # No, random is better for exploration. But we want to be fast.
    rng = np.random.default_rng(42)
    
    # We prioritize lower ranks.
    # Rank 27 is the theoretical lower bound.
    # Let's try ranks from 27 to 40 first.
    # If time remains, go higher.
    
    ranks_to_try = list(range(27, 41))
    
    # Allocate time: 
    # Give more time to lower ranks.
    # Total time ~110s.
    # Let's split time proportionally or evenly.
    # Even split is simple.
    
    n_ranks = len(ranks_to_try)
    time_per_rank = 110.0 / n_ranks
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        # Don't spend more than remaining time
        t_budget = min(time_per_rank, remaining_time - 1.0)
        
        if t_budget < 1.0:
            break
            
        result = _search_rank(n, m, p, r, t_budget, rng)
        
        if result is not None:
            U, V, W, rank_found = result
            # Final verification
            if _verify_einsum(U, V, W, n, m, p, tol=1e-9):
                return {"U": U, "V": V, "W": W}
                
    # If we didn't find a solution in the lower range, try higher ranks with remaining time
    # Or just return the best we found? The task requires a valid decomposition.
    # If we fail to find one, we should return a known valid one (schoolbook).
    
    # Fallback: Schoolbook decomposition (Rank 48)
    U_sch = np.zeros((n*m, n*m*p))
    V_sch = np.zeros((m*p, n*m*p))
    W_sch = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m + j, col] = 1
                V_sch[j*p + k, col] = 1
                W_sch[k*n + i, col] = 1
                col += 1
                
    if _verify_einsum(U_sch, V_sch, W_sch, n, m, p):
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
        
    # This should never happen
    raise RuntimeError("Failed to find valid decomposition")

# EVOLVE-BLOCK-END
