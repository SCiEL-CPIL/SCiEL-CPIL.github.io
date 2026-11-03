"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # This is the known best solution so far.
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # r=7: A02 B20 -> C00
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # r=11: A00 B02 -> C02
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U17[3, 14] = 1
    V17[2, 14] = 1
    W17[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U17[4, 15] = 1
    V17[5, 15] = 1
    W17[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U17, V17, W17, n, m, p, rank_17):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U17, V17, W17
    best_rank = rank_17
    
    # Attempt to find rank 16 via local optimization
    # We initialize a rank-16 solution by taking the rank-17 solution and 
    # projecting it or randomly dropping a term, then optimizing.
    
    target_rank = 16
    np.random.seed(42)
    
    # Strategy: Try to find a rank-16 solution by minimizing the error
    # of the tensor decomposition.
    # Variables: U, V, W flattened.
    # This is a high-dimensional non-convex problem.
    # We will try a few random initializations derived from the rank-17 solution.
    
    def error_func(params):
        if time.time() > deadline:
            return 1e10
        U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank + m*p*target_rank], params[n*m*target_rank + m*p*target_rank:]
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for r in range(target_rank):
                            s += W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        err += (s - target) ** 2
        return err
    
    # Initialize with rank-17 solution, dropping one column randomly
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Randomly select a column to drop from the rank-17 solution
        drop_idx = np.random.randint(0, rank_17)
        keep_idxs = [i for i in range(rank_17) if i != drop_idx]
        
        U_init = U17[:, keep_idxs]
        V_init = V17[:, keep_idxs]
        W_init = W17[:, keep_idxs]
        
        # Add small noise to escape local minima
        noise_scale = 0.1
        U_init = U_init + np.random.normal(0, noise_scale, U_init.shape)
        V_init = V_init + np.random.normal(0, noise_scale, V_init.shape)
        W_init = W_init + np.random.normal(0, noise_scale, W_init.shape)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use Nelder-Mead for local optimization
        # This is slow, so we limit iterations.
        try:
            res = minimize(error_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*target_rank].reshape((n*m, target_rank))
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape((p*n, target_rank))
                
                # Verify exactness
                if _verify(U_opt, V_opt, W_opt, n, m, p, target_rank):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break
        except Exception:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
