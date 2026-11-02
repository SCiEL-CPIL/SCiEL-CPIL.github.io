import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for A (2x2)
    A_factors = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    B_factors = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    for b in range(8):
        if time.time() > deadline:
            break
            
        jL = 2 * b
        jR = 2 * b + 1
        
        for s in range(7):
            col = b * 7 + s
            
            for idx, val in enumerate(A_factors[s]):
                U[idx, col] = val
                
            V_indices = [0 * 16 + jL, 0 * 16 + jR, 1 * 16 + jL, 1 * 16 + jR]
            for idx, val in enumerate(B_factors[s]):
                V[V_indices[idx], col] = val
                
            w_C0L = jL * 2 + 0
            w_C0R = jR * 2 + 0
            w_C1L = jL * 2 + 1
            w_C1R = jR * 2 + 1
            
            if s == 0:
                W[w_C0L, col] += 1
                W[w_C1R, col] += 1
            elif s == 1:
                W[w_C1L, col] += 1
                W[w_C1R, col] -= 1
            elif s == 2:
                W[w_C0R, col] += 1
                W[w_C1R, col] += 1
            elif s == 3:
                W[w_C0L, col] += 1
                W[w_C1L, col] += 1
            elif s == 4:
                W[w_C0L, col] -= 1
                W[w_C0R, col] += 1
            elif s == 5:
                W[w_C1R, col] += 1
            elif s == 6:
                W[w_C0L, col] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
