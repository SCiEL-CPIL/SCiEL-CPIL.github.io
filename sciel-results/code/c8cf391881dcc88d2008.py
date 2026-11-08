import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin
    
    # Target tensor T[i, l, k] = 1 for all valid indices
    # C[i, k] = sum_l A[i, l] B[l, k]
    T = np.ones((n, m, p))
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U_school[i * m + l, col] = 1
                V_school[l * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # We know rank <= 54.
    # Let's try 42 to 53.
    
    for rank in range(42, 54):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W randomly
        np.random.seed(42 + rank)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline:
                break
                
            # Solve for V
            # For each l, k: M[l, k] is 2 x rank, t[l, k] is 2
            V_new = np.zeros((m * p, rank))
            for l in range(m):
                for k in range(p):
                    # M[i, c] = U[i, l, c] * W[k, i, c]
                    # U is (n*m, rank). U[i*m + l, c]
                    # W is (p*n, rank). W[k*n + i, c]
                    M_vk = np.zeros((n, rank))
                    for i in range(n):
                        M_vk[i, :] = U[i * m + l, :] * W[k * n + i, :]
                    # t[i] = T[i, l, k] = 1
                    t_vk = np.ones(n)
                    # Solve M_vk v = t_vk
                    # v is rank x 1
                    try:
                        v_vk, _, _, _ = np.linalg.lstsq(M_vk, t_vk, rcond=None)
                        V_new[l * p + k, :] = v_vk
                    except:
                        pass
                        
            V = V_new
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for l in range(m):
                    # M[k, c] = V[l, k, c] * W[k, i, c]
                    M_uk = np.zeros((p, rank))
                    for k in range(p):
                        M_uk[k, :] = V[l * p + k, :] * W[k * n + i, :]
                    t_uk = np.ones(p)
                    try:
                        u_uk, _, _, _ = np.linalg.lstsq(M_uk, t_uk, rcond=None)
                        U_new[i * m + l, :] = u_uk
                    except:
                        pass
            U = U_new
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # M[l, c] = U[i, l, c] * V[l, k, c]
                    M_wk = np.zeros((m, rank))
                    for l in range(m):
                        M_wk[l, :] = U[i * m + l, :] * V[l * p + k, :]
                    t_wk = np.ones(m)
                    try:
                        w_wk, _, _, _ = np.linalg.lstsq(M_wk, t_wk, rcond=None)
                        W_new[k * n + i, :] = w_wk
                    except:
                        pass
            W = W_new
            
        # Check residual
        # Compute T_reconstructed
        # T_recon[i, l, k] = sum_c U[i, l, c] V[l, k, c] W[k, i, c]
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            # u_c is n x m
            u_c = U[:, c].reshape(n, m)
            # v_c is m x p
            v_c = V[:, c].reshape(m, p)
            # w_c is p x n
            w_c = W[:, c].reshape(p, n)
            # T_recon += u_c[i, l] * v_c[l, k] * w_c[k, i]
            # This is element-wise product sum
            # Loop over i, l, k
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        T_recon[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                        
        residual = np.linalg.norm(T - T_recon)
        
        if residual < 1e-8:
            # Try to round to half-integers
            # Check if 2*U is close to integer
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            T_exact = np.zeros((n, m, p))
            for c in range(rank):
                u_c = U_round[:, c].reshape(n, m)
                v_c = V_round[:, c].reshape(m, p)
                w_c = W_round[:, c].reshape(p, n)
                for i in range(n):
                    for l in range(m):
                        for k in range(p):
                            T_exact[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                            
            if np.allclose(T, T_exact, atol=1e-9):
                # Found a valid decomposition
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
    return best_result
