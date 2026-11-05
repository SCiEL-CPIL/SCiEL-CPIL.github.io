import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    rank_school = n * m * p
    U_school = np.zeros((dim_U, rank_school))
    V_school = np.zeros((dim_V, rank_school))
    W_school = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110
    
    for R in range(30, 61):
        if time.time() > deadline:
            break
            
        np.random.seed(R)
        U = np.random.randn(dim_U, R)
        V = np.random.randn(dim_V, R)
        W = np.random.randn(dim_W, R)
        
        for it in range(50):
            if time.time() > deadline:
                break
                
            X = (U[:, :, None] * V[None, :, :])
            X_flat = X.reshape(-1, R)
            T_flat = T.reshape(-1, dim_W)
            
            for w_idx in range(dim_W):
                b = T_flat[:, w_idx]
                w_sol, _, _, _ = np.linalg.lstsq(X_flat, b, rcond=None)
                W[w_idx, :] = w_sol
                
            Y = (V[:, :, None] * W[None, :, :])
            Y_flat = Y.reshape(-1, R)
            
            for u_idx in range(dim_U):
                b = T[u_idx, :, :].flatten()
                u_sol, _, _, _ = np.linalg.lstsq(Y_flat, b, rcond=None)
                U[u_idx, :] = u_sol
                
            Z = (U[:, :, None] * W[None, :, :])
            Z_flat = Z.reshape(-1, R)
            
            for v_idx in range(dim_V):
                b = T[:, v_idx, :].flatten()
                v_sol, _, _, _ = np.linalg.lstsq(Z_flat, b, rcond=None)
                V[v_idx, :] = v_sol
        
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        T_recon = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
        error = np.linalg.norm(T_recon - T)
        
        if error < 1e-9:
            best_result = {"U": U_snap, "V": V_snap, "W": W_snap}
            best_rank = R
            break
            
    U_list = [[float(val) for val in row] for row in best_result["U"]]
    V_list = [[float(val) for val in row] for row in best_result["V"]]
    W_list = [[float(val) for val in row] for row in best_result["W"]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
