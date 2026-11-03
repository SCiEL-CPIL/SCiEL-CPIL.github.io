import time
import random
import math
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_global = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def get_tensor(n, m, p):
        """Return the multiplication tensor as a dictionary of non-zero entries."""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[(a, b, c)] = 1
        return T

    def validate(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is exact using NumPy."""
        if not U or not V or not W:
            return False
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Result[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
        # Step 1: Contract U and V over k -> Temp[a, b, k]
        Temp = np.einsum('ak,bk->abk', U_arr, V_arr)
        # Step 2: Contract Temp and W over k -> Result[a, b, c]
        Result = np.einsum('abk,ck->abc', Temp, W_arr)
        
        # Build target tensor
        Target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    Target[a, b, c] = 1
        
        return np.allclose(Result, Target, atol=tol)

    def schoolbook_decomposition(n, m, p):
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

    def try_find_decomposition(n, m, p, target_rank, time_limit=30):
        """Try to find a decomposition of rank target_rank."""
        start_time = time.time()
        
        def objective(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Fast error calculation
            Temp = np.einsum('ak,bk->abk', U, V)
            Result = np.einsum('abk,ck->abc', Temp, W)
            
            Target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        Target[a, b, c] = 1
                        
            err = np.sum((Result - Target) ** 2)
            err += 0.01 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return err

        def discrete_polish(U, V, W, max_iters=200):
            """Refine U, V, W to half-integers using local search."""
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U, V, W
                
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                # Randomly pick one of the matrices and one element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    idx = np.random.randint(n*m*target_rank)
                    i, c = np.unravel_index(idx, (n*m, target_rank))
                    old_val = U[i, c]
                    U[i, c] = old_val + 0.5 * np.random.choice([-1, 1])
                    if validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return U, V, W
                    U[i, c] = old_val
                elif mat_idx == 1:
                    idx = np.random.randint(m*p*target_rank)
                    i, c = np.unravel_index(idx, (m*p, target_rank))
                    old_val = V[i, c]
                    V[i, c] = old_val + 0.5 * np.random.choice([-1, 1])
                    if validate(V.tolist(), U.tolist(), W.tolist(), n, m, p): # Note: validate expects U, V, W order
                        return U, V, W
                    V[i, c] = old_val
                else:
                    idx = np.random.randint(p*n*target_rank)
                    i, c = np.unravel_index(idx, (p*n, target_rank))
                    old_val = W[i, c]
                    W[i, c] = old_val + 0.5 * np.random.choice([-1, 1])
                    if validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return U, V, W
                    W[i, c] = old_val
            return None, None, None

        best_sol = None
        best_err = float('inf')
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_sol = res.x
                
                if best_err < 1e-3:
                    U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Attempt discrete polish
                    U_p, V_p, W_p = discrete_polish(U_sol, V_sol, W_sol, max_iters=100)
                    if U_p is not None:
                        return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}

        return None

    # Strategy:
    # 1. Try to find rank 23 for 3x3x3
    # 2. If found, construct rank 46 for 3x3x6
    # 3. If not, try to find rank < 54 for 3x3x6 directly

    # First, try to find rank 23 for 3x3x3
    res_333 = try_find_decomposition(3, 3, 3, 23, time_limit=40)
    
    if res_333 is not None:
        # Construct rank 46 for 3x3x6
        U_333 = res_333['U']
        V_333 = res_333['V']
        W_333 = res_333['W']
        
        rank_333 = len(U_333[0])
        
        U_336 = [[0] * (2 * rank_333) for _ in range(9)]
        V_336 = [[0] * (2 * rank_333) for _ in range(18)]
        W_336 = [[0] * (2 * rank_333) for _ in range(18)]
        
        for c in range(rank_333):
            # Block 1
            for i in range(9):
                U_336[i][c] = U_333[i][c]
            for j in range(3):
                for k in range(3):
                    V_336[j*3 + k][c] = V_333[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_336[k*3 + i][c] = W_333[k*3 + i][c]
            
            # Block 2
            for i in range(9):
                U_336[i][c + rank_333] = U_333[i][c]
            for j in range(3):
                for k in range(3):
                    V_336[(j*3 + 3) + k][c + rank_333] = V_333[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_336[(k*3 + 3) + i][c + rank_333] = W_333[k*3 + i][c]
        
        if validate(U_336, V_336, W_336, 3, 3, 6):
            return {"U": U_336, "V": V_336, "W": W_336}
        
    # Search for 3x3x6 directly
    # Try ranks 46, 47, ..., 53
    for r in range(46, 54):
        if time.time() > deadline - 10:
            break
        res = try_find_decomposition(3, 3, 6, r, time_limit=10)
        if res is not None:
            return res
            
    # Fallback
    return schoolbook_decomposition(3, 3, 6)
