"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute index arrays for vectorized tensor evaluation
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    U_idx = i_vals * m + j_vals
    V_idx = j_vals * p + k_vals
    W_idx = k_vals * n + i_vals
    
    np.random.seed(42)
    
    # Try to find a low-rank decomposition
    for rank in range(30, 97):
        for trial in range(20):
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            best_err = float('inf')
            for it in range(50):
                U_sel = U[U_idx, :]
                V_sel = V[V_idx, :]
                W_sel = W[W_idx, :]
                vals = (U_sel * V_sel * W_sel).sum(axis=1)
                diff = vals - 1.0
                err = float(np.sum(diff * diff))
                
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        break
                
                # Compute gradients efficiently
                diff_3d = diff.reshape(n, m, p)
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(rank):
                    U_r = U[:, r].reshape(n, m)
                    V_r = V[:, r].reshape(m, p)
                    W_r = W[:, r].reshape(p, n)
                    
                    grad_U_r = (diff_3d * V_r[None, :, :] * W_r.T[:, :, None]).sum(axis=2)
                    grad_U[:, r] = 2 * grad_U_r.flatten()
                    
                    grad_V_r = (diff_3d * U_r[:, :, None] * W_r.T[:, None, :]).sum(axis=0)
                    grad_V[:, r] = 2 * grad_V_r.flatten()
                    
                    grad_W_r = (diff_3d * U_r[:, :, None] * V_r[None, :, :]).sum(axis=1)
                    grad_W[:, r] = 2 * grad_W_r.T.flatten()
                
                lr = 0.02
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                U = np.clip(U, -3, 3)
                V = np.clip(V, -3, 3)
                W = np.clip(W, -3, 3)
            
            if best_err < 1e-5:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
