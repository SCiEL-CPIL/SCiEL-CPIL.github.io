import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    full_rank = n * m * p
    U_best = [[0] * full_rank for _ in range(n * m)]
    V_best = [[0] * full_rank for _ in range(m * p)]
    W_best = [[0] * full_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_best, "V": V_best, "W": W_best}
    
    # Target tensor
    target_tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_tensor[u_idx, v_idx, w_idx] = 1.0
                
    T_flat = target_tensor.reshape(n * m * m * p, p * n)
    
    def try_find_decomposition(rank):
        if time.time() - start_time > deadline - 5:
            return None
            
        trials = 0
        max_trials = 500
        
        while trials < max_trials:
            if time.time() - start_time > deadline - 5:
                return None
            trials += 1
            
            U_rand = np.zeros((n * m, rank))
            V_rand = np.zeros((m * p, rank))
            
            for a in range(rank):
                # Sparse random
                idx_u = np.random.choice(n * m, size=max(1, int(n * m * 0.5)), replace=False)
                idx_v = np.random.choice(m * p, size=max(1, int(m * p * 0.5)), replace=False)
                
                U_rand[idx_u, a] = np.random.choice([-1, 1], size=len(idx_u))
                V_rand[idx_v, a] = np.random.choice([-1, 1], size=len(idx_v))
            
            B_matrix = np.zeros((n * m * m * p, rank))
            for a in range(rank):
                M_a = np.outer(U_rand[:, a], V_rand[:, a])
                B_matrix[:, a] = M_a.flatten()
                
            try:
                W_trans, residuals, rank_b, s = np.linalg.lstsq(B_matrix, T_flat, rcond=None)
                residual = np.linalg.norm(B_matrix @ W_trans - T_flat)
                
                if residual < 1e-8:
                    W_sol = W_trans.T
                    # Check for integer/half-integer
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-7):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_sol.tolist()}
            except:
                pass
        
        return None

    # Try rank 44
    res = try_find_decomposition(44)
    if res:
        return res

    # Try rank 45
    if time.time() - start_time < deadline - 10:
        res = try_find_decomposition(45)
        if res:
            return res

    return best_result
