import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    M = np.zeros((n*m, m*p, p*n))
    nz_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                M[idx1, idx2, idx3] = 1.0
                nz_idx.append((idx1, idx2, idx3))
    nz_idx = np.array(nz_idx)
    
    M_flat = M.reshape(n*m, m*p * p*n)
    M_T_flat = M.T.reshape(m*p, n*m*p*n)
    M_P_flat = M.reshape(n*m*m*p, p*n).T
    
    best_result = None
    best_rank = None
    
    lower_bound = max(n*p, m*n, m*p)
    
    for rank in range(lower_bound, 73):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 20 if rank <= 28 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 1000 + restart
            np.random.seed(seed)
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            converged = False
            prev_residual = np.inf
            
            for iteration in range(60):
                if time.time() > deadline - 5:
                    break
                
                T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
                T_flat = T.reshape(m*p*p*n, rank)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(T_flat.T, M_flat.T, rcond=None)
                    U_new = U_new.T
                except:
                    break
                
                T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
                T2_flat = T2.reshape(n*m*p*n, rank)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(T2_flat.T, M_T_flat.T, rcond=None)
                    V_new = V_new.T
                except:
                    break
                
                T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
                T3_flat = T3.reshape(n*m*m*p, rank)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(T3_flat.T, M_P_flat.T, rcond=None)
                    W_new = W_new.T
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                i_vals = nz_idx[:, 0]
                j_vals = nz_idx[:, 1]
                k_vals = nz_idx[:, 2]
                U_vals = U[i_vals, :]
                V_vals = V[j_vals, :]
                W_vals = W[k_vals, :]
                tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                residual = np.abs(tensor_vals - 1.0).max()
                
                if residual < 1e-10:
                    converged = True
                    break
                
                if residual > prev_residual * 0.998:
                    if iteration > 8:
                        break
                prev_residual = residual
            
            if converged:
                for polish_iter in range(15):
                    if time.time() > deadline - 3:
                        break
                    
                    T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
                    T_flat = T.reshape(m*p*p*n, rank)
                    try:
                        U_new, _, _, _ = np.linalg.lstsq(T_flat.T, M_flat.T, rcond=None)
                        U_new = U_new.T
                    except:
                        break
                    
                    T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
                    T2_flat = T2.reshape(n*m*p*n, rank)
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(T2_flat.T, M_T_flat.T, rcond=None)
                        V_new = V_new.T
                    except:
                        break
                    
                    T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
                    T3_flat = T3.reshape(n*m*m*p, rank)
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(T3_flat.T, M_P_flat.T, rcond=None)
                        W_new = W_new.T
                    except:
                        break
                    
                    U, V, W = U_new, V_new, W_new
                    
                    i_vals = nz_idx[:, 0]
                    j_vals = nz_idx[:, 1]
                    k_vals = nz_idx[:, 2]
                    U_vals = U[i_vals, :]
                    V_vals = V[j_vals, :]
                    W_vals = W[k_vals, :]
                    tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                    residual = np.abs(tensor_vals - 1.0).max()
                    
                    if residual < 1e-12:
                        break
                
                T_check = (U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :])
                final_residual = np.abs(np.sum(T_check, axis=3) - M).max()
                
                if final_residual < 1e-10:
                    best_rank = rank
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
