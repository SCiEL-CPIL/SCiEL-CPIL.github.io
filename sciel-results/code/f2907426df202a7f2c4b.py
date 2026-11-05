import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    C = A @ B
    C_target = np.zeros(p*n)
    for i in range(n):
        for k in range(p):
            C_target[k*n + i] = C[i, k]
            
    u = U_np @ A_flat
    v = V_np @ B_flat
    c = W_np @ (u * v)
    
    if not np.allclose(c, C_target, atol=1e-5):
        return False
    return True
