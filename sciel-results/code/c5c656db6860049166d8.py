import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140.0
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
        rank = len(U[0]) if U else 0
        if rank == 0: return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    # Strategy:
    # 1. Directly search for a rank-minimal decomposition for the full 3x3x5 problem.
    # 2. Use a parameterized search where we fix the structure to be sparse (mostly 0, 1, -1)
    #    but allow some continuous values to be optimized, or use a combinatorial search
    #    for the support and then solve for the weights.
    # 3. Given the time limit, we will try to find a decomposition of rank R < 45.
    #    We know R >= 27 (lower bound).
    #    Let's try R from 27 upwards.
    #    For each R, we formulate it as a system of equations.
    #    However, the problem is non-linear.
    #    
    #    Alternative: Use the known decomposition for 3x3x3 (rank 23 is not known to exist,
    #    schoolbook is 27, lower bound 19).
    #    Actually, for 3x3x3, the minimal rank is not known. It is between 19 and 23 (Strassen 1980 showed 23).
    #    Wait, Strassen's algorithm for 3x3x3 has rank 23? No, Strassen is for 2x2x2.
    #    For 3x3x3, the best known upper bound is 23 (from some constructions).
    #    If we use a rank 23 decomposition for 3x3x3, and add 18 for 3x3x2, total is 41.
    #    If we use rank 27 (schoolbook) for 3x3x3, total is 45.
    #    
    #    Let's check if we can find a better decomposition for 3x3x5 directly.
    #    We will try to solve for a rank R decomposition using a heuristic search.
    #    
    #    We will try ranks 27 to 35.
    #    For each rank, we will try to find a solution using a local search or gradient descent
    #    on the residual.
    
    def residuals_full(params, R):
        # params: flattened U (9*R), V (15*R), W (15*R)
        U_flat = params[:9 * R].reshape(9, R)
        V_flat = params[9 * R:9 * R + 15 * R].reshape(15, R)
        W_flat = params[9 * R + 15 * R:].reshape(15, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    # We will attempt to find a decomposition with rank R.
    # We start with a random initialization and use least squares.
    # Since the problem is highly underdetermined for R > 45 and overdetermined for R < 45,
    # we are looking for a specific sparse solution.
    
    # A more effective strategy might be to exploit the block structure.
    # Let's try to find a decomposition of rank 40, 39, ... down to 27.
    
    best_solution = None
    best_rank = 45
    
    # Fallback
    U_school, V_school, W_school, _ = make_schoolbook(3, 3, 5)
    
    # Try to improve upon schoolbook
    # We'll use a randomized restart strategy with least_squares.
    
    for target_rank in range(27, 45):
        if time.time() > deadline - 20:
            break
        
        # We only need to try a few restarts for each rank to save time
        max_restarts = 5 if target_rank > 35 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.default_rng(123 + restart * 100 + target_rank)
            
            # Sparse initialization might help
            # Initialize with small random values
            U_init = rng.normal(size=(9, target_rank)) * 0.1
            V_init = rng.normal(size=(15, target_rank)) * 0.1
            W_init = rng.normal(size=(15, target_rank)) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    lambda p: np.sum(residuals_full(p, target_rank) ** 2),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                # Check if we found a solution
                res_val = res.fun
                if res_val < 1e-12:
                    U_sol = res.x[:9 * target_rank].reshape(9, target_rank)
                    V_sol = res.x[9 * target_rank:9 * target_rank + 15 * target_rank].reshape(15, target_rank)
                    W_sol = res.x[9 * target_rank + 15 * target_rank:].reshape(15, target_rank)
                    
                    # Try to round to half-integers or integers
                    # First try rounding to nearest 0.5
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break # Found a valid solution for this rank, try lower ranks? No, we go up.
                        # Wait, we are iterating from 27 up. So if we find one at 27, that's best.
                        # We should break out of the inner loop and return immediately?
                        # Yes, because we want the minimal rank.
                        return best_solution
                        
            except Exception:
                pass

    # If we didn't find a better solution, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
