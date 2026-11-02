import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition
    # Rank is n * m * p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per task description W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                # Check time
                if time.time() > deadline - safety_margin:
                    break
            if time.time() > deadline - safety_margin:
                break
        if time.time() > deadline - safety_margin:
            break
            
    # If we broke early due to time, the decomposition is invalid (incomplete).
    # However, for 3*4*7=84 iterations, it will never take 140 seconds.
    # So this check is just for safety.
    
    return {"U": U, "V": V, "W": W}
