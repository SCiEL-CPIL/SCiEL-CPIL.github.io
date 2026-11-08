import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Search for lower rank
    for r in range(20, rank_sb):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
            
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            # Precompute contributions for fast updates
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Let's store T
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            error = np.sum((T - 1) ** 2)
            
            if error < 1e-9:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
            
            # Local search
            # We will try to reduce error by changing one variable at a time
            # We keep track of T to update it incrementally
            
            improved = True
            stall_count = 0
            max_stall = 50
            max_iters = 5000
            iter_count = 0
            
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                if time.time() > deadline:
                    break
                    
                # Pick a random variable type and index
                var_type = np.random.randint(3)
                
                if var_type == 0: # Change U[i,j,c]
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    c = np.random.randint(r)
                    old_val = U_3d[i, j, c]
                    # Update T for all k
                    # Delta T[i,j,k] = (new_val - old_val) * V[j,k,c] * W[k,i,c]
                    delta_U = np.array([-1, -0.5, 0, 0.5, 1])
                    for new_val in delta_U:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        # V[j,:,c] is shape (p,)
                        # W[:,i,c] is shape (p,)
                        # Delta T[i,j,:] = diff * (V[j,:,c] * W[:,i,c])
                        delta_T = diff * (V_3d[j, :, c] * W_3d[:, i, c])
                        # T[i,j,:] is shape (p,)
                        T_new = T[i, j, :] + delta_T
                        # Compute new error for this slice
                        # Old error slice: sum (T[i,j,k]-1)^2
                        # New error slice: sum (T_new[k]-1)^2
                        old_err_slice = np.sum((T[i, j, :] - 1) ** 2)
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        if new_err_slice < old_err_slice - 1e-12:
                            T[i, j, :] = T_new
                            U_3d[i, j, c] = new_val
                            error += new_err_slice - old_err_slice
                            improved = True
                            break
                            
                elif var_type == 1: # Change V[j,k,c]
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    c = np.random.randint(r)
                    old_val = V_3d[j, k, c]
                    delta_V = np.array([-1, -0.5, 0, 0.5, 1])
                    for new_val in delta_V:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        # Delta T[i,j,k] = U[i,j,c] * diff * W[k,i,c]
                        # T[:,j,k] is shape (n,)
                        # U[:,j,c] is shape (n,)
                        # W[k,:,c] is shape (n,)
                        delta_T = (U_3d[:, j, c] * W_3d[k, :, c]) * diff
                        T_new = T[:, j, k] + delta_T
                        old_err_slice = np.sum((T[:, j, k] - 1) ** 2)
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        if new_err_slice < old_err_slice - 1e-12:
                            T[:, j, k] = T_new
                            V_3d[j, k, c] = new_val
                            error += new_err_slice - old_err_slice
                            improved = True
                            break
                            
                else: # Change W[k,i,c]
                    k = np.random.randint(p)
                    i = np.random.randint(n)
                    c = np.random.randint(r)
                    old_val = W_3d[k, i, c]
                    delta_W = np.array([-1, -0.5, 0, 0.5, 1])
                    for new_val in delta_W:
                        if new_val == old_val: continue
                        diff = new_val - old_val
                        if diff == 0: continue
                        # Delta T[i,j,k] = U[i,j,c] * V[j,k,c] * diff
                        # T[i,:,k] is shape (m,)
                        # U[i,:,c] is shape (m,)
                        # V[:,k,c] is shape (m,)
                        delta_T = (U_3d[i, :, c] * V_3d[:, k, c]) * diff
                        T_new = T[i, :, k] + delta_T
                        old_err_slice = np.sum((T[i, :, k] - 1) ** 2)
                        new_err_slice = np.sum((T_new - 1) ** 2)
                        if new_err_slice < old_err_slice - 1e-12:
                            T[i, :, k] = T_new
                            W_3d[k, i, c] = new_val
                            error += new_err_slice - old_err_slice
                            improved = True
                            break
                            
                if not improved:
                    stall_count += 1
                    if stall_count > max_stall:
                        break
                else:
                    stall_count = 0
                    
            if error < 1e-9:
                best_rank = r
                best_U = U_3d.flatten().reshape(n*m, r)
                best_V = V_3d.flatten().reshape(m*p, r)
                best_W = W_3d.flatten().reshape(p*n, r)
                found = True
                break
        
        if found:
            break

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
