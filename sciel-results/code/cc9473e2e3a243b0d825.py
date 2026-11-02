import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    # Use numpy for faster verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1
    # We can compute this efficiently
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try random search for lower ranks
    # For 4x4x9, we can try ranks from a reasonable minimum up
    # The tensor rank for 4x4x9 is likely much lower than 144
    # Try ranks from 20 upward (a reasonable lower bound)
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            U = [[random.choice([-1,0,1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1,0,1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1,0,1]) for _ in range(r)] for _ in range(p*n)]
            if verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
        if best_rank == r:
            break
    
    return best
