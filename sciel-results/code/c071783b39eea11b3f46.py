"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 independent 2x2x2 Strassen blocks
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # A vector: [A00, A01, A10, A11]
    # B vector: [B00, B01, B10, B11]
    # C vector: [C00, C10, C01, C11] (matches W[k*n+i] convention)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1 = (a+d)(e+h)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2 = (c+d)e
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3 = a(f-h)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4 = d(g-e)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5 = (a+b)h
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6 = (c-a)(e+f)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7 = (b-d)(g+h)
    ]
    
    col = 0
    for t in range(p // 2):
        if time.time() > deadline:
            break
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, flat index i*m+j
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10, for block t columns 2t, 2t+1
            # B[j, 2t]   -> V[j*p + 2t]
            # B[j, 2t+1] -> V[j*p + 2t + 1]
            V[0 * p + 2 * t][col] = b_vec[0]
            V[0 * p + 2 * t + 1][col] = b_vec[1]
            V[1 * p + 2 * t][col] = b_vec[2]
            V[1 * p + 2 * t + 1][col] = b_vec[3]
            
            # W: C is 2x10, convention W[k*n + i]
            # C[0, 2t]   -> W[2t * n + 0] = W[4t]
            # C[1, 2t]   -> W[2t * n + 1] = W[4t+1]
            # C[0, 2t+1] -> W[(2t+1) * n + 0] = W[4t+2]
            # C[1, 2t+1] -> W[(2t+1) * n + 1] = W[4t+3]
            W[2 * t * n + 0][col] = c_vec[0]
            W[2 * t * n + 1][col] = c_vec[1]
            W[(2 * t + 1) * n + 0][col] = c_vec[2]
            W[(2 * t + 1) * n + 1][col] = c_vec[3]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
