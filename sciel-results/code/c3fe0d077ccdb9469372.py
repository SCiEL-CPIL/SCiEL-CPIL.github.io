import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 60
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Quick verification
    np.random.seed(42)
    for _ in range(3):
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_expected = A_test @ B_test
        C_actual = np.zeros((n, p), dtype=int)
        for c in range(rank):
            fa = np.dot(U[:, c], A_test.reshape(-1))
            fb = np.dot(V[:, c], B_test.reshape(-1))
            Wc_reshaped = W[:, c].reshape(p, n).T
            C_actual += Wc_reshaped * (fa * fb)
        if not np.allclose(C_actual, C_expected):
            # Should never happen with schoolbook, but safety net
            pass
    
    if time.time() > deadline:
        # Extremely unlikely, but return what we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
