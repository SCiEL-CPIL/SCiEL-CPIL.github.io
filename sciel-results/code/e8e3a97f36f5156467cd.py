import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Schoolbook fallback
    def get_schoolbook():
        n, m, p = 2, 3, 8
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

    # Target: 2x3x2 tensor
    n2, m2, p2 = 2, 3, 2
    T_target = np.ones((n2, m2, p2))
    
    def solve_232(rank):
        if time.time() > deadline - 20:
            return None
            
        # Random restarts
        for restart in range(20):
            if time.time() > deadline - 20:
                return None
                
            np.random.seed(restart * 100 + rank)
            U = np.random.randn(n2 * m2, rank)
            V = np.random.randn(m2 * p2, rank)
            W = np.random.randn(p2 * n2, rank)
            
            # ALS
            for it in range(200):
                if time.time() > deadline - 20:
                    return None
                    
                # Update U
                # T = sum U[:, r] outer(V[:, r], W[:, r])
                # Linear in U
                A = np.zeros((n2 * m2 * p2, n2 * m2 * rank))
                for r in range(rank):
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                idx_t = (i * m2 + j) * p2 + k
                                idx_u = (i * m2 + j) * rank + r
                                A[idx_t, idx_u] = V[j * p2 + k, r] * W[k * n2 + i, r]
                t_vec = T_target.flatten()
                U_flat, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                U = U_flat.reshape(n2 * m2, rank)
                
                # Update V
                A = np.zeros((n2 * m2 * p2, m2 * p2 * rank))
                for r in range(rank):
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                idx_t = (i * m2 + j) * p2 + k
                                idx_v = (j * p2 + k) * rank + r
                                A[idx_t, idx_v] = U[i * m2 + j, r] * W[k * n2 + i, r]
                V_flat, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                V = V_flat.reshape(m2 * p2, rank)
                
                # Update W
                A = np.zeros((n2 * m2 * p2, p2 * n2 * rank))
                for r in range(rank):
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                idx_t = (i * m2 + j) * p2 + k
                                idx_w = (k * n2 + i) * rank + r
                                A[idx_t, idx_w] = U[i * m2 + j, r] * V[j * p2 + k, r]
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                W = W_flat.reshape(p2 * n2, rank)
                
                # Check convergence
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_recon, T_target, atol=1e-6):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    T_recon_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.allclose(T_recon_r, T_target, atol=1e-6):
                        return U_r, V_r, W_r
                        
        return None

    best_result = get_schoolbook()
    
    for rank in [9, 10, 11]:
        sol = solve_232(rank)
        if sol:
            U232, V232, W232 = sol
            R = rank
            
            # Construct 2,3,8
            U238 = np.zeros((6, 4 * R))
            V238 = np.zeros((24, 4 * R))
            W238 = np.zeros((16, 4 * R))
            
            for r in range(R):
                for r_B in range(4):
                    col = r * 4 + r_B
                    # U
                    U238[:, col] = U232[:, r]
                    # V
                    for j in range(3):
                        for k_A in range(2):
                            for k_B in range(4):
                                row = (j * 2 + k_A) * 4 + k_B
                                V238[row, col] = V232[j * 2 + k_A, r]
                    # W
                    for k_A in range(2):
                        for i in range(2):
                            for k_B in range(4):
                                row = (k_A * 4 + k_B) * 2 + i
                                if k_B == r_B:
                                    W238[row, col] = W232[k_A * 2 + i, r]
                                else:
                                    W238[row, col] = 0
                                    
            # Verify
            T_full = np.einsum('ir,jr,kr->ijk', U238, V238, W238)
            T_check = np.ones((2, 3, 8))
            if np.allclose(T_full, T_check, atol=1e-6):
                return {
                    "U": U238.tolist(),
                    "V": V238.tolist(),
                    "W": W238.tolist()
                }
                
    return best_result
