import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution
from fractions import Fraction

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-8)

    def try_round_and_verify(U_num, V_num, W_num, denom=2):
        """Try rounding to multiples of 1/denom and verify."""
        U_rnd = np.round(U_num * denom) / denom
        V_rnd = np.round(V_num * denom) / denom
        W_rnd = np.round(W_num * denom) / denom
        if verify(U_rnd, V_rnd, W_rnd):
            return U_rnd, V_rnd, W_rnd, True
        return None, None, None, False

    def polish(U_num, V_num, W_num, max_iter=5000):
        """High-precision polish pass using tighter tolerances."""
        nm, mp, pn = n*m, m*p, p*n
        rank = U_num.shape[1]
        params0 = np.concatenate([U_num.flatten(), V_num.flatten(), W_num.flatten()])
        
        def objective(params):
            U_flat = params[:nm*rank]
            V_flat = params[nm*rank:nm*rank + mp*rank]
            W_flat = params[nm*rank + mp*rank:]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            T_prime = compute_tensor(U, V, W)
            diff = T_prime - T_target
            return np.sum(diff**2)
        
        try:
            # Use L-BFGS-B for final high-precision convergence
            res = minimize(objective, params0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < 1e-12:
                U_flat = res.x[:nm*rank]
                V_flat = res.x[nm*rank:nm*rank + mp*rank]
                W_flat = res.x[nm*rank + mp*rank:]
                return (U_flat.reshape(nm, rank), V_flat.reshape(mp, rank), W_flat.reshape(pn, rank))
        except Exception:
            pass
        return U_num, V_num, W_num

    np.random.seed(42)
    
    found = False
    
    # We focus on rank 10 first, as it is the most likely minimal rank.
    # If rank 10 fails, we try rank 11.
    target_ranks = [10, 11]
    
    for target_rank in target_ranks:
        if found or time.time() > deadline - 20.0:
            break
        
        nm, mp, pn = n*m, m*p, p*n
        total_params = (nm + mp + pn) * target_rank
        
        # Define bounds for DE. 
        # Tensor entries are typically small. 
        # Using a wide range to ensure global search capability.
        bounds = [(-10, 10)] * total_params
        
        def objective(params):
            U_flat = params[:nm*target_rank]
            V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
            W_flat = params[nm*target_rank + mp*target_rank:]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            T_prime = compute_tensor(U, V, W)
            diff = T_prime - T_target
            return np.sum(diff**2)

        # Strategy: Global search using Differential Evolution
        # DE is stochastic and global, so it can escape local minima that trap L-BFGS.
        # We allocate most of the time to this step.
        
        # Calculate time available for DE
        time_for_de = min(100.0, deadline - time.time() - 10.0)
        
        # DE parameters
        # maxiter is a soft limit, we rely on the timeout callback
        # We use 'best1bin' crossover which is standard
        
        # To save time, we can reduce the population size if time is tight, 
        # but for 36-54 parameters, a population of 50-100 is manageable.
        popsize = 50
        
        try:
            # We use a wrapper to check time
            # scipy DE doesn't have a direct timeout, so we rely on maxiter and tol
            # A small tol helps it stop early if it converges.
            
            # For rank 10, we are looking for a global minimum.
            # If DE finds a solution with error < 1e-4, we polish it.
            
            # Note: DE can be slow. We limit maxiter based on time.
            # Rough estimate: 1000 iterations with popsize 50 is ~50k function evals.
            # Each eval is small. Should be fast enough.
            
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=2000, 
                tol=1e-8, 
                popsize=popsize,
                mutation=(0.5, 1.0),
                crossover=0.9,
                seed=42,
                polish=False # We do our own polish
            )
            
            if res.fun < 1e-4:
                U_flat = res.x[:nm*target_rank]
                V_flat = res.x[nm*target_rank:nm*target_rank + mp*target_rank]
                W_flat = res.x[nm*target_rank + mp*target_rank:]
                U_num = U_flat.reshape(nm, target_rank)
                V_num = V_flat.reshape(mp, target_rank)
                W_num = W_flat.reshape(pn, target_rank)
                
                # Polish for higher precision
                if time.time() < deadline - 5.0:
                    U_num, V_num, W_num = polish(U_num, V_num, W_num)
                
                # Try different rounding denominators
                # The problem states "integer/half-integer entries"
                # So we should primarily check denom=2. 
                # But sometimes the "exact" solution might be integer (denom=1).
                for denom in [2, 1, 4, 8]:
                    U_rnd, V_rnd, W_rnd, ok = try_round_and_verify(U_num, V_num, W_num, denom)
                    if ok:
                        best_U, best_V, best_W = U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist()
                        best_rank = target_rank
                        found = True
                        break
                    if time.time() > deadline - 5.0:
                        break
                if found:
                    break
                    
        except Exception as e:
            # Fallback to local search if DE fails or times out unexpectedly
            pass

        # Fallback: If DE didn't find it quickly, try a few local optimizations from random starts
        if not found and time.time() < deadline - 10.0:
            max_attempts = 20
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                # Random initialization
                scale = 1.0
                params0 = np.random.randn(total_params) * scale
                
                def local_objective(params):
                    U_flat = params[:nm*target_rank]
                    V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
                    W_flat = params[nm*target_rank + mp*target_rank:]
                    
                    U = U_flat.reshape(nm, target_rank)
                    V = V_flat.reshape(mp, target_rank)
                    W = W_flat.reshape(pn, target_rank)
                    
                    T_prime = compute_tensor(U, V, W)
                    diff = T_prime - T_target
                    return np.sum(diff**2)

                try:
                    res = minimize(local_objective, params0, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                    if res.fun < 1e-8:
                        U_flat = res.x[:nm*target_rank]
                        V_flat = res.x[nm*target_rank:nm*target_rank + mp*target_rank]
                        W_flat = res.x[nm*target_rank + mp*target_rank:]
                        U_num = U_flat.reshape(nm, target_rank)
                        V_num = V_flat.reshape(mp, target_rank)
                        W_num = W_flat.reshape(pn, target_rank)
                        
                        U_num, V_num, W_num = polish(U_num, V_num, W_num)
                        
                        for denom in [2, 1, 4, 8]:
                            U_rnd, V_rnd, W_rnd, ok = try_round_and_verify(U_num, V_num, W_num, denom)
                            if ok:
                                best_U, best_V, best_W = U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist()
                                best_rank = target_rank
                                found = True
                                break
                            if time.time() > deadline - 3.0:
                                break
                        if found:
                            break
                except Exception:
                    continue

    return {"U": best_U, "V": best_V, "W": best_W}
