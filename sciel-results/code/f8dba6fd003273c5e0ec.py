"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 construction:
    # 5 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using Schoolbook (rank 4)
    # Total 5*7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00 -> C10+1, C11-1
    # M3: A00(B01-B11) -> C01+1, C11+1
    # M4: A11(B10-B00) -> C00+1, C10+1
    # M5: (A00+A01)B11 -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01) -> C11+1
    # M7: (A01-A11)(B10+B11) -> C00-1
    
    # Local indices for A: 0:00, 1:01, 2:10, 3:11
    # Local indices for B: 0:00, 1:01, 2:10, 3:11
    # Local indices for C: 0:00, 1:01, 2:10, 3:11 (k*n+i -> 0*2+0, 1*2+0, 0*2+1, 1*2+1)
    # Wait, W index is k*n+i.
    # For 2x2 block, k is local 0 or 1.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, 1, -1],# M2: C10+1, C11-1
        [0, 1, 0, 1], # M3: C01+1, C11+1
        [1, 0, 1, 0], # M4: C00+1, C10+1
        [-1, 1, 0, 0],# M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [-1, 0, 0, 0] # M7: C00-1
    ]
    
    # Fill U, V, W for 5 Strassen blocks (cols 0-1, 2-3, 4-5, 6-7, 8-9)
    for b in range(5):
        r_offset = b * 7
        for r in range(7):
            col = r_offset + r
            
            # U
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    U[local_idx][col] = strassen_U[r][local_idx]
            
            # V
            # Local B indices:
            # 0: 0, 2b
            # 1: 0, 2b+1
            # 2: 1, 2b
            # 3: 1, 2b+1
            for j in range(2):
                for k_local in range(2):
                    local_idx = j * 2 + k_local
                    global_k = 2 * b + k_local
                    global_idx = j * p + global_k
                    V[global_idx][col] = strassen_V[r][local_idx]
            
            # W
            # Local C indices (k_local, i):
            # 0: 0, 0
            # 1: 0, 1
            # 2: 1, 0
            # 3: 1, 1
            # Global W index: global_k * n + i
            for k_local in range(2):
                for i in range(2):
                    local_idx = k_local * 2 + i
                    global_k = 2 * b + k_local
                    global_idx = global_k * n + i
                    W[global_idx][col] = strassen_W[r][local_idx]
    
    # Fill U, V, W for last column (col 10)
    r_offset = 35
    # T1: A00 B0,10 -> C0,10
    U[0][r_offset] = 1
    V[0 * p + 10][r_offset] = 1
    W[10 * n + 0][r_offset] = 1
    
    # T2: A01 B1,10 -> C0,10
    U[1][r_offset + 1] = 1
    V[1 * p + 10][r_offset + 1] = 1
    W[10 * n + 0][r_offset + 1] = 1
    
    # T3: A10 B0,10 -> C1,10
    U[2][r_offset + 2] = 1
    V[0 * p + 10][r_offset + 2] = 1
    W[10 * n + 1][r_offset + 2] = 1
    
    # T4: A11 B1,10 -> C1,10
    U[3][r_offset + 3] = 1
    V[1 * p + 10][r_offset + 3] = 1
    W[10 * n + 1][r_offset + 3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
