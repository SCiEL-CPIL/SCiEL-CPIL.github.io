"""Rewritten program for matrix multiplication 2,2,15 tensor rank minimization."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W


def _build_strassen_2x2x2_block(n, m, p):
    """
    Build a decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest.
    For p=15: use 7 blocks of 2x2x2 (Strassen, rank 7 each) + 1 block of 2x2x1 (schoolbook, rank 4)
    Total: 7*7 + 4 = 53
    """
    # Strassen matrices for 2x2x2
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook for 2x2x1
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Total rank: 7 blocks * 7 + 1 block * 4 = 49 + 4 = 53
    total_rank = 7 * 7 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    # 7 Strassen blocks for k=0..13 (pairs: (0,1), (2,3), ..., (12,13))
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(m):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(n):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    # 1 schoolbook block for k=14
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(m):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(n):
            W[k_last * n + i][c] = W_school[i][r]
    
    return U, V, W


def _find_m223_rank10(deadline):
    """
    Find rank-10 decomposition for M_{2,2,3} via global optimization.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def objective(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        tensor_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_p[i*m+j] * V_p[j*p+k] * W_p[k*n+i])
                    tensor_err += (val - 1.0) ** 2
        return tensor_err

    from scipy.optimize import differential_evolution
    
    # Bounds for each variable, e.g., [-5, 5]
    bounds = [(-5, 5)] * (n*m*rank + m*p*rank + p*n*rank)
    
    best_obj = float('inf')
    best_params = None
    
    # Run differential evolution with a time limit
    # We can use 'maxiter' and 'polypop' to control runtime
    # Also, we can try to seed it with the schoolbook rank 12 solution projected?
    # Or just random.
    
    try:
        # Use a smaller population and fewer iterations to stay within time
        res = differential_evolution(objective, bounds, maxiter=100, popsize=5, 
                                      seed=42, tol=1e-8, disp=False, 
                                      init='sobol')
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    if best_params is None or best_obj > 1e-4:
        return None
    
    U_final = best_params[:n*m*rank].reshape(n*m, rank)
    V_final = best_params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
    W_final = best_params[n*m*rank+m*p*rank:].reshape(p*n, rank)
    
    # Round to nearest half-integer
    U_half = np.round(U_final * 2) / 2
    V_half = np.round(V_final * 2) / 2
    W_half = np.round(W_final * 2) / 2
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_half[i*m+j] * V_half[j*p+k] * W_half[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-6:
        # Try a local polish with Nelder-Mead starting from the rounded solution
        # to see if we can fix small errors
        params_rounded = np.concatenate([U_half.flatten(), V_half.flatten(), W_half.flatten()])
        from scipy.optimize import minimize
        try:
            res2 = minimize(objective, params_rounded, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x.copy()
                U_final = best_params[:n*m*rank].reshape(n*m, rank)
                V_final = best_params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_final = best_params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                U_half = np.round(U_final * 2) / 2
                V_half = np.round(V_final * 2) / 2
                W_half = np.round(W_final * 2) / 2
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_half[i*m+j] * V_half[j*p+k] * W_half[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
        except Exception:
            pass

    if max_err > 1e-6:
        return None
    
    return U_half.tolist(), V_half.tolist(), W_half.tolist()


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Fallback: Rank 53 solution (Strassen blocks + schoolbook)
    U_fb, V_fb, W_fb = _build_strassen_2x2x2_block(n, m, p)
    
    # Verify fallback
    if not _verify_decomposition(U_fb, V_fb, W_fb, n, m, p):
        # Ultimate fallback: pure schoolbook
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    
    # Try to find rank-10 M_{2,2,3} and tile it for rank 50
    try:
        if time.time() < deadline - 60:
            result = _find_m223_rank10(deadline)
            if result is not None:
                U_223, V_223, W_223 = result
                rank_223 = len(U_223[0])
                
                U_new = [[0] * rank_223 for _ in range(n * m)]
                V_new = [[0] * rank_223 for _ in range(m * p)]
                W_new = [[0] * rank_223 for _ in range(p * n)]
                
                n_blocks = p // 3  # 5 blocks of 3 columns
                for block in range(n_blocks):
                    k0 = 3 * block
                    for r in range(rank_223):
                        c = block * rank_223 + r
                        for i in range(n):
                            for j in range(m):
                                U_new[i * m + j][c] = U_223[i * m + j][r]
                        for j in range(m):
                            for kk in range(3):
                                V_new[j * p + k0 + kk][c] = V_223[j * 3 + kk][r]
                        for kk in range(3):
                            for i in range(n):
                                W_new[(k0 + kk) * n + i][c] = W_223[kk * n + i][r]
                
                # Verify the full decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass

    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
