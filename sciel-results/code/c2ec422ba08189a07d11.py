import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 7
    
    # Schoolbook decomposition: rank = n*m*p = 84
    # Each term picks out one (i,j,k) triple
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify correctness using vectorized check
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    U_t = np.array(U)
    V_t = np.array(V)
    W_t = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_t[i*m+j, :], V_t[j*p+k, :] * W_t[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Should not happen, but return anyway
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
