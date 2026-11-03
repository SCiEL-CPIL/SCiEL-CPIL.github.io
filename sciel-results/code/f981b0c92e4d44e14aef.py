import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_fast(U, V, W, n, m, p):
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    for i in range(n):
        W_i = W_r[:, i, :]
        U_i = U_r[i, :, :]
        X = np.einsum('jkr,kr->jkr', V_r, W_i)
        T_i = np.einsum('jr,jkr->jk', U_i, X)
        if not np.allclose(T_i, 1.0, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best = _schoolbook()
    
    # Try to find a better rank
    # We know 96 is achievable.
    # Try 90, 85, 80
    for target_rank in [90, 85, 80]:
        if time.time() > deadline:
            break
            
        # Random search
        found = False
        for _ in range(100):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, size=(n * m, target_rank))
            V = np.random.randint(-1, 2, size=(m * p, target_rank))
            W = np.random.randint(-1, 2, size=(p * n, target_rank))
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
                
        if found:
            break
            
    return best
