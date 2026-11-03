import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # Try ranks from 44 down to 30
    # Schoolbook is 45
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    best_result = None
    
    for R in range(44, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        num_restarts = 5 if R > 40 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Random initialization
            np.random.seed(42 + restart + R)
            U = np.random.rand(dim_U, R)
            V = np.random.rand(dim_V, R)
            W = np.random.rand(dim_W, R)
            
            # ALS iterations
            max_iter = 100
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, R))
                        for k in range(p):
                            idx_v = j * p + k
                            idx_w = k * n + i
                            A[k, :] = V[idx_v, :] * W[idx_w, :]
                        # Solve A @ u = 1
                        # Use least squares
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[idx_u, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        # B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        B = np.zeros((n, R))
                        for i in range(n):
                            idx_u = i * m + j
                            idx_w = k * n + i
                            B[i, :] = U[idx_u, :] * W[idx_w, :]
                        # Solve B @ v = 1
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[idx_v, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        # C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        C = np.zeros((m, R))
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            C[j, :] = U[idx_u, :] * V[idx_v, :]
                        # Solve C @ w = 1
                        w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[idx_w, :] = w
                
                # Check residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-6:
                    # Try to round
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded
                    res_r = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.dot(U_r[idx_u, :], V_r[idx_v, :] * W_r[idx_w, :])
                                res_r += (val - 1.0) ** 2
                                
                    if res_r < 1e-9:
                        # Success
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                    
                    # If rounding failed, maybe continue optimizing or break
                    # If we are close, maybe just return the float version? 
                    # No, task requires integer/half-integer.
                    # So we must find an exact one.
                    # If we found a float solution, we might be stuck.
                    # Let's break out of iterations for this restart.
                    break
                    
            # End ALS iterations
            
        # End restarts
        
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
