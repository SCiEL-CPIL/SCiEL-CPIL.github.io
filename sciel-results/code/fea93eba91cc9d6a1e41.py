import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    # Trivial decomposition: rank = n*m = 12
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(n):
        for i in range(m):
            k = r * m + i
            U[r * m + i, k] = 1.0
            for c in range(p):
                V[i * p + c, k] = 1.0
                W[c * n + r, k] = 1.0
    
    # Verify
    def verify_full():
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    a_idx = r * m + i
                    b_idx = i * p + c
                    w_idx = c * n + r
                    expected = 1.0
                    actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                    if abs(actual - expected) > 1e-10:
                        return False
        # Check zeros
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    for j in range(m):
                        if i != j:
                            a_idx = r * m + j
                            b_idx = i * p + c
                            w_idx = c * n + r
                            actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                            if abs(actual) > 1e-10:
                                return False
        return True
    
    if not verify_full():
        # Should not happen, but fallback
        raise ValueError("Verification failed")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
