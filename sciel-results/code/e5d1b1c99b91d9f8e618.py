import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p
    
    # Schoolbook
    U_school = np.zeros((n * m, r_school), dtype=int)
    V_school = np.zeros((m * p, r_school), dtype=int)
    W_school = np.zeros((p * n, r_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
                
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = r_school
    
    # Try to find lower rank
    # Start with r = 50
    for r in [50, 60, 70, 80, 90, 100]:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        U = np.random.randint(-1, 2, (n * m, r))
        V = np.random.randint(-1, 2, (m * p, r))
        W = np.random.randint(-1, 2, (p * n, r))
        
        # Compute T_comp
        T_comp = np.zeros((n * m, m * p, p * n), dtype=int)
        for c in range(r):
            T_comp += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
        error = np.sum((T_comp - T_target) ** 2)
        
        if error == 0:
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
            
        # Local Search
        no_improve = 0
        while time.time() < deadline - 5:
            if error == 0:
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
                
            # Pick random update
            mat_type = random.choice([0, 1, 2])
            if mat_type == 0:
                mat = U
                rows = n * m
                # Delta T = delta * outer(e_i, V[:,j], W[:,j])
                # We need V[:,j] and W[:,j]
            elif mat_type == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
                
            i = random.randint(0, rows - 1)
            j = random.randint(0, r - 1)
            delta = random.choice([-1, 1])
            
            old_val = mat[i, j]
            mat[i, j] = old_val + delta
            
            # Compute Delta T
            if mat_type == 0:
                # e_i is one-hot
                # We can just update the slice T_comp[i, :, :]
                # Delta T[i, a, b] = delta * V[a, j] * W[b, j]
                # T_comp[i, :, :] += delta * np.outer(V[:, j], W[:, j])
                T_comp[i, :, :] += delta * np.outer(V[:, j], W[:, j])
            elif mat_type == 1:
                # Delta T[a, i, b] = delta * U[a, j] * W[b, j]
                # T_comp[:, i, :] += delta * np.outer(U[:, j], W[:, j])
                T_comp[:, i, :] += delta * np.outer(U[:, j], W[:, j])
            else:
                # Delta T[a, b, i] = delta * U[a, j] * V[b, j]
                # T_comp[:, :, i] += delta * np.outer(U[:, j], V[:, j])
                T_comp[:, :, i] += delta * np.outer(U[:, j], V[:, j])
                
            new_error = np.sum((T_comp - T_target) ** 2)
            
            if new_error < error:
                error = new_error
                no_improve = 0
            else:
                # Revert
                mat[i, j] = old_val
                if mat_type == 0:
                    T_comp[i, :, :] -= delta * np.outer(V[:, j], W[:, j])
                elif mat_type == 1:
                    T_comp[:, i, :] -= delta * np.outer(U[:, j], W[:, j])
                else:
                    T_comp[:, :, i] -= delta * np.outer(U[:, j], V[:, j])
                no_improve += 1
                if no_improve > 10000:
                    break
                    
    return best_result
