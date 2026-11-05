import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Target Tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
            
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.int64)
        V = np.zeros((dim_v, rank), dtype=np.int64)
        W = np.zeros((dim_w, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    time_limit = 115.0
    
    # Try ranks from 85 to 95
    # Lower is better, so try lower ranks first
    for rank in range(85, 96):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts with Simulated Annealing
        max_restarts = 100
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init with integers -2, -1, 0, 1, 2
            # Using smaller range might be better for convergence
            U = np.random.choice([-2, -1, 0, 1, 2], size=(dim_u, rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(dim_v, rank))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(dim_w, rank))
            
            # Compute initial error
            def compute_error(U, V, W):
                T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
                return np.sum((T - T_hat) ** 2)
            
            current_error = compute_error(U, V, W)
            if current_error == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Simulated Annealing
            T_curr = 10.0
            T_min = 0.01
            alpha = 0.999
            steps = 5000
            
            for step in range(steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick random element to change
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dim1, dim2 = dim_u, rank
                elif mat_idx == 1:
                    mat = V
                    dim1, dim2 = dim_v, rank
                else:
                    mat = W
                    dim1, dim2 = dim_w, rank
                    
                idx1 = np.random.randint(0, dim1)
                idx2 = np.random.randint(0, dim2)
                
                old_val = mat[idx1, idx2]
                new_val = old_val + np.random.choice([-1, 1])
                
                # Bound values to avoid explosion
                if new_val > 5: new_val = 5
                if new_val < -5: new_val = -5
                
                mat[idx1, idx2] = new_val
                new_error = compute_error(U, V, W)
                
                delta_e = new_error - current_error
                
                if delta_e < 0:
                    current_error = new_error
                    if current_error == 0:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    # Accept with probability
                    if T_curr > T_min:
                        p = np.exp(-delta_e / T_curr)
                        if np.random.random() < p:
                            current_error = new_error
                        else:
                            mat[idx1, idx2] = old_val
                    else:
                        mat[idx1, idx2] = old_val
                        
                T_curr *= alpha

    return get_schoolbook()
