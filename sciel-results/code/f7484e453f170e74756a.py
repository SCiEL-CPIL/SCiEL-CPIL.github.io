import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    rank_school = n * m * p
    
    # Schoolbook solution
    U_init = np.zeros((dim_U, rank_school))
    V_init = np.zeros((dim_V, rank_school))
    W_init = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_init[u_idx, col] = 1
                V_init[v_idx, col] = 1
                W_init[w_idx, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_init, V_init, W_init

    start_time = time.time()
    max_time = 140  # Safety margin

    ranks_to_try = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for k in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        restarts = 20 if k <= 27 else 5
        
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 1000 + k * 100 + restart
            np.random.seed(seed_val)
            
            # Random dense initialization
            U = np.random.randn(dim_U, k)
            V = np.random.randn(dim_V, k)
            W = np.random.randn(dim_W, k)
            
            # ALS iterations
            for it in range(50):
                # Update U
                # M_VW = Khatri-Rao(V, W) shape (dim_V * dim_W, k)
                M_VW = np.einsum('ir,jr->ijr', V, W).reshape(dim_V * dim_W, k)
                # T_1 shape (dim_U, dim_V * dim_W)
                T_1 = T.reshape(dim_U, dim_V * dim_W)
                U = T_1 @ np.linalg.pinv(M_VW.T)
                
                # Update V
                M_UW = np.einsum('ir,jr->ijr', U, W).reshape(dim_U * dim_W, k)
                T_2 = T.reshape(dim_V, dim_U * dim_W)
                V = T_2 @ np.linalg.pinv(M_UW.T)
                
                # Update W
                M_UV = np.einsum('ir,jr->ijr', U, V).reshape(dim_U * dim_V, k)
                T_3 = T.reshape(dim_W, dim_U * dim_V)
                W = T_3 @ np.linalg.pinv(M_UV.T)
                
            # Round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify
            Rec = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
            if np.allclose(Rec, T, atol=1e-9):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
