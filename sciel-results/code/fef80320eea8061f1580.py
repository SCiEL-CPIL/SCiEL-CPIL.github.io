import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _compute_error(U, V, W, n, m, p):
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    D = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.sum((D - 1.0)**2)

def _compute_gradient(U, V, W, n, m, p):
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # Error = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
    # Let D[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # grad_U[i,j,c] = 2 * (D[i,j,k]-1) * V[j,k,c] * W[k,i,c] summed over k
    # grad_V[j,k,c] = 2 * (D[i,j,k]-1) * U[i,j,c] * W[k,i,c] summed over i
    # grad_W[k,i,c] = 2 * (D[i,j,k]-1) * U[i,j,c] * V[j,k,c] summed over j
    
    D = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    err = D - 1.0
    
    # grad_U[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
    # = 2 * einsum('ijk,jkc,kic->ijc', err, V_r, W_r)
    gU = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, V_r, W_r)
    
    # grad_V[j,k,c] = 2 * sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
    # = 2 * einsum('ijk,ijc,kic->jkc', err, U_r, W_r)
    gV = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, U_r, W_r)
    
    # grad_W[k,i,c] = 2 * sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
    # = 2 * einsum('ijk,ijc,jkc->kic', err, U_r, V_r)
    gW = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, U_r, V_r)
    
    return gU.reshape(-1, r), gV.reshape(-1, r), gW.reshape(-1, r)

def construct():
    n, m, p = DIMENSIONS
    dim = n * m * p
    deadline = time.time() + 110
    
    # Schoolbook
    U_school = np.zeros((n*m, dim), dtype=np.float64)
    V_school = np.zeros((m*p, dim), dtype=np.float64)
    W_school = np.zeros((p*n, dim), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = dim
    best_err = _compute_error(best_U, best_V, best_W, n, m, p)
    
    def try_reduce_rank(U_init, V_init, W_init, current_rank, target_rank, time_budget):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        r = current_rank
        start_time = time.time()
        
        while r > target_rank:
            if time.time() - start_time > time_budget:
                return None, None, None, r
            
            success = False
            attempts = 0
            max_attempts = 15
            attempt_start = time.time()
            
            while attempts < max_attempts and time.time() - attempt_start < 3.0:
                c = np.random.randint(r)
                
                U_save = U.copy()
                V_save = V.copy()
                W_save = W.copy()
                
                U[:, c] = 0.0
                V[:, c] = 0.0
                W[:, c] = 0.0
                
                active_cols = [i for i in range(r) if i != c]
                
                best_err_local = _compute_error(U, V, W, n, m, p)
                
                # Gradient descent with adaptive step size
                opt_start = time.time()
                step = 0.05
                prev_err = best_err_local
                
                while time.time() - opt_start < 2.0:
                    gU, gV, gW = _compute_gradient(U, V, W, n, m, p)
                    
                    # Only update active columns
                    for c_act in active_cols:
                        U[:, c_act] -= step * gU[:, c_act]
                        V[:, c_act] -= step * gV[:, c_act]
                        W[:, c_act] -= step * gW[:, c_act]
                    
                    curr_err = _compute_error(U, V, W, n, m, p)
                    
                    if curr_err < prev_err:
                        # Good step, maybe increase slightly
                        step = min(step * 1.2, 0.5)
                        prev_err = curr_err
                        if curr_err < best_err_local:
                            best_err_local = curr_err
                    else:
                        # Bad step, decrease and revert
                        step = step * 0.7
                        # Revert
                        gU, gV, gW = _compute_gradient(U, V, W, n, m, p)
                        for c_act in active_cols:
                            U[:, c_act] += step * gU[:, c_act]
                            V[:, c_act] += step * gV[:, c_act]
                            W[:, c_act] += step * gW[:, c_act]
                        # Try smaller step
                        step = step * 0.5
                        if step < 1e-6:
                            break
                    
                    if best_err_local < 1e-8:
                        break
                
                if best_err_local < 1e-6:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    r -= 1
                    success = True
                    break
                else:
                    U, V, W = U_save, V_save, W_save
                    attempts += 1
            
            if not success:
                return None, None, None, r
        
        return U, V, W, r
    
    # Try to reduce to 74
    if time.time() < deadline - 25:
        res = try_reduce_rank(best_U, best_V, best_W, best_rank, 74, 45)
        if res[3] == 74:
            best_U, best_V, best_W, best_rank = res
            best_err = _compute_error(best_U, best_V, best_W, n, m, p)
    
    # Try to reduce to 73
    if best_rank == 74 and time.time() < deadline - 15:
        res = try_reduce_rank(best_U, best_V, best_W, best_rank, 73, 35)
        if res[3] == 73:
            best_U, best_V, best_W, best_rank = res
            best_err = _compute_error(best_U, best_V, best_W, n, m, p)
    
    # Try to reduce to 72
    if best_rank == 73 and time.time() < deadline - 10:
        res = try_reduce_rank(best_U, best_V, best_W, best_rank, 72, 25)
        if res[3] == 72:
            best_U, best_V, best_W, best_rank = res
            best_err = _compute_error(best_U, best_V, best_W, n, m, p)
    
    # Round and verify best
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    err = _compute_error(U_out, V_out, W_out, n, m, p)
    if err > 1e-9:
        U_out = np.round(U_school * 2) / 2
        V_out = np.round(V_school * 2) / 2
        W_out = np.round(W_school * 2) / 2
        best_rank = dim

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
