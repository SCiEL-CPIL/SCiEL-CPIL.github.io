import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0
    
    # Try ranks from 18 to 23
    for rank in range(18, 24):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        # Estimate restarts
        # 1000 steps * 10us = 10ms per restart
        # 100 restarts per second
        max_restarts = int(time_left * 100)
        
        if max_restarts < 10:
            continue
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-3, 4, (n*m, rank))
            V = np.random.randint(-3, 4, (m*p, rank))
            W = np.random.randint(-3, 4, (p*n, rank))
            
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute error
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                error = np.sum((T - 1)**2)
                
                if error == 0:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Pick a random move
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    old_val = U[row, col]
                    new_val = old_val + random.choice([-1, 1])
                    if abs(new_val) > 6: continue
                    U[row, col] = new_val
                elif mat_idx == 1:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    old_val = V[row, col]
                    new_val = old_val + random.choice([-1, 1])
                    if abs(new_val) > 6: continue
                    V[row, col] = new_val
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    old_val = W[row, col]
                    new_val = old_val + random.choice([-1, 1])
                    if abs(new_val) > 6: continue
                    W[row, col] = new_val
                
                # Compute new error
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                new_error = np.sum((T - 1)**2)
                
                # Accept if better
                if new_error <= error:
                    error = new_error
                else:
                    # Revert
                    if mat_idx == 0:
                        U[row, col] = old_val
                    elif mat_idx == 1:
                        V[row, col] = old_val
                    else:
                        W[row, col] = old_val
            
            # Polish step
            if error < 5:
                # Try to fix the remaining error
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    # Find the index with max error
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    W_r = W.reshape(p, n, rank)
                    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    err_vec = T - 1
                    max_idx = np.unravel_index(np.argmax(np.abs(err_vec)), err_vec.shape)
                    i, j, k = max_idx
                    # Try to change U[i, j, r], V[j, k, r], W[k, i, r]
                    # This is complex, so we just do more random steps
                    pass
        
    # Fallback
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
