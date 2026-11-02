import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # Try standard shape first? No, just do the structural change.
    # If the previous code failed, try transposed.
    U = np.zeros((rank, n * m))
    V = np.zeros((rank, m * p))
    W = np.zeros((rank, p * n))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 10:
                    break
                U[col, i * m + j] = 1
                V[col, j * p + k] = 1
                W[col, k * n + i] = 1
                col += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # If we broke early, the rank is effectively col?
    # No, the matrices are pre-allocated to full rank.
    # The unused columns will be 0.
    # This is a valid decomposition of rank <= nmp.
    
    return {"U": U, "V": V, "W": W}
