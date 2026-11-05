"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before the 120s limit

    # Build the target tensor
    # T[(i,j), (j,k), (i,k)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    schoolbook_rank = n * m * p

    # Start with schoolbook as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    best_result = (U_sch, V_sch, W_sch)
    best_rank = schoolbook_rank

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W give an exact decomposition of T."""
        rank = U.shape[1]
        approx = np.zeros_like(T)
        for r in range(rank):
            approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T, approx, atol=tol)

    def try_round_and_verify(U, V, W, tol=1e-10):
        """Round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r, tol):
            return U_r, V_r, W_r
        return None

    def structured_init_3x3x5(target_rank):
        """
        Initialize using structure: treat as 3 independent 3x3x1 problems
        plus cross-coupling terms.
        """
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Strategy: use rank-19 3x3x3 decomposition for each of 3 column-blocks,
        # but share some terms to reduce total rank.
        # Known: 3x3x3 has rank 19 (Strassen-like) or 23 (schoolbook).
        # For 3x3x5, we can potentially do better than 3*19=57.
        
        # Let's try a structured approach:
        # Assign columns 0..target_rank-1 to specific (i,j,k) patterns
        # with some sharing.
        
        # Simple structured init: cycle through (i,j,k) with wrapping
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if idx < target_rank:
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
                        idx += 1
        return U, V, W

    def random_init(target_rank, scale=0.5):
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.randn(p*n, target_rank) * scale
        return U, V, W

    def objective_factory(T, n, m, p, target_rank):
        """Create objective function for given rank."""
        def objective(params):
            U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
            
            return np.sum((T - approx)**2)
        return objective

    def try_optimize(target_rank, U_init, V_init, W_init, max_iter=300):
        """Run optimization and return result if successful."""
        params_init = np.concatenate([
            U_init.flatten(),
            V_init.flatten(),
            W_init.flatten()
        ])
        
        try:
            from scipy.optimize import minimize
            obj = objective_factory(T, n, m, p, target_rank)
            result = minimize(obj, params_init, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'maxfun': max_iter*2})
            
            if result.fun < 1e-8:
                U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                rounded = try_round_and_verify(U_r, V_r, W_r)
                if rounded is not None:
                    return rounded
        except Exception:
            pass
        return None

    # Search from lower ranks upward
    # Lower bound for 3x3x5: at least max(9, 15, 15) = 15
    # Known results suggest rank ~14-19 might be achievable
    lower_bound = 14
    upper_bound = min(schoolbook_rank, 30)
    
    for target_rank in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try structured initialization first
        if time.time() < deadline - 20:
            U_s, V_s, W_s = structured_init_3x3x5(target_rank)
            res = try_optimize(target_rank, U_s, V_s, W_s, max_iter=200)
            if res is not None:
                U_r, V_r, W_r = res
                best_result = (U_r, V_r, W_r)
                best_rank = target_rank
                found = True
                break
        
        # Try random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            U_rnd, V_rnd, W_rnd = random_init(target_rank, scale=1.0)
            res = try_optimize(target_rank, U_rnd, V_rnd, W_rnd, max_iter=150)
            if res is not None:
                U_r, V_r, W_r = res
                best_result = (U_r, V_r, W_r)
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found working rank, try to go lower
            continue
    
    # Convert best result to the required format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
