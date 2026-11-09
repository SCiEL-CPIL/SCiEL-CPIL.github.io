import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
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

    # Reshape T for ALS
    # T_U: (dim_U, dim_V * dim_W)
    T_U = T.reshape(dim_U, dim_V * dim_W)
    # T_V: (dim_V, dim_U * dim_W)
    # Note: T is indexed T[u, v, w].
    # T_V[v, u*dim_W + w] = T[u, v, w]
    T_V = np.zeros((dim_V, dim_U * dim_W))
    for u in range(dim_U):
        for v in range(dim_V):
            for w in range(dim_W):
                T_V[v, u * dim_W + w] = T[u, v, w]
                
    # T_W: (dim_W, dim_U * dim_V)
    T_W = np.zeros((dim_W, dim_U * dim_V))
    for u in range(dim_U):
        for v in range(dim_V):
            for w in range(dim_W):
                T_W[w, u * dim_V + v] = T[u, v, w]

    start_time = time.time()
    max_time = 115 # Safety margin

    # Try ranks from 25 up to schoolbook
    # Based on directive, 32 is a strong candidate.
    # We try 25..35 to find the minimal one.
    ranks_to_try = list(range(25, rank_school + 1))
    
    # Prioritize 32? No, we want minimal.
    # But if 25-31 are impossible, we waste time.
    # However, ALS is fast.
    
    for r in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        # Number of restarts
        # Give more restarts to likely ranks
        restarts = 50 if r in [30, 31, 32, 33, 34] else 10
        
        found = False
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 1000 + r * 100 + restart
            np.random.seed(seed_val)
            
            # Initialize
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # ALS
            for it in range(100):
                if time.time() - start_time > max_time:
                    break
                    
                # Update U
                # B_VW: (dim_V * dim_W, r)
                # B_VW[v*dim_W + w, c] = V[v, c] * W[w, c]
                B_VW = (V[:, :, None] * W[None, :, :]).reshape(dim_V * dim_W, r)
                # Solve U * B_VW^T = T_U
                # U = T_U * B_VW^T * pinv(B_VW * B_VW^T)
                try:
                    U_new = np.linalg.lstsq(B_VW.T, T_U, rcond=None)[0]
                except:
                    break
                if np.allclose(U, U_new):
                    break
                U = U_new
                
                # Update V
                # B_UW: (dim_U * dim_W, r)
                # B_UW[u*dim_W + w, c] = U[u, c] * W[w, c]
                B_UW = (U[:, :, None] * W[None, :, :]).reshape(dim_U * dim_W, r)
                try:
                    V_new = np.linalg.lstsq(B_UW.T, T_V, rcond=None)[0]
                except:
                    break
                if np.allclose(V, V_new):
                    break
                V = V_new
                
                # Update W
                # B_UV: (dim_U * dim_V, r)
                # B_UV[u*dim_V + v, c] = U[u, c] * V[v, c]
                B_UV = (U[:, :, None] * V[None, :, :]).reshape(dim_U * dim_V, r)
                try:
                    W_new = np.linalg.lstsq(B_UV.T, T_W, rcond=None)[0]
                except:
                    break
                if np.allclose(W, W_new):
                    break
                W = W_new
                
            # Check error
            Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(Rec - T)
            
            if err < 1e-9:
                # Round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
    # Fallback
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
