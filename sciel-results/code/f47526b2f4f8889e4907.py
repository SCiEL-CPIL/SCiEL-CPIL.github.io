"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part + Rank 16 for 5th column
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen 4x4x4 decomposition
    # Define the 2x2x2 Strassen coefficients
    strassen_U = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00+A01
        [-1, 0, 1, 0],# P6: A10-A00
        [0, 1, 0, -1] # P7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1],# P3: B01-B11
        [-1, 0, 1, 0],# P4: B10-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # P1: C00+C11
        [0, 1, 0, 0], # P2: C10
        [0, 0, 1, 0], # P3: C01
        [1, 1, 0, 0], # P4: C00+C10
        [-1, 0, 1, 0],# P5: -C00+C01
        [0, 0, 0, 1], # P6: C11
        [1, 0, 0, 0]  # P7: C00
    ]
    
    col = 0
    for r in range(2):
        for s in range(2):
            for t in range(2):
                for q in range(7):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                    for i in range(2):
                        for j in range(2):
                            u_val = strassen_U[q][i * 2 + j]
                            if u_val != 0:
                                u_idx = (2 * r + i) * m + (2 * s + j)
                                U[u_idx][col] += u_val
                                
                            v_val = strassen_V[q][i * 2 + j]
                            if v_val != 0:
                                v_idx = (2 * s + i) * p + (2 * t + j)
                                V[v_idx][col] += v_val
                                
                            w_val = strassen_W[q][i * 2 + j]
                            if w_val != 0:
                                w_idx = (2 * r + j) * n + (2 * t + i)
                                W[w_idx][col] += w_val
                    col += 1
                    
    # 2. Schoolbook for 5th column (k=4)
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
