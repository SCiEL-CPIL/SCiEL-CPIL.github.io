import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Split m into m1=2, m2=3
    m1 = 2
    m2 = 3
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # B1 is split into 3 blocks of 2x2: B1_0, B1_1, B1_2
    # Each block uses Strassen (rank 7)
    # Total rank for Part 1: 3 * 7 = 21
    
    # Part 2: A2 (2x3) * B2 (3x6)
    # Use schoolbook (rank 2*3*6 = 36)
    # Total rank for Part 2: 36
    
    total_rank = 21 + 36
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 2x2 blocks ---
    # Strassen decomposition for 2x2 matrix multiplication
    # Indices for 2x2: i in 0..1, j in 0..1, k in 0..1
    # U is 4x7, V is 4x7, W is 4x7
    
    # Strassen coefficients
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01+y11)
    # M5 = (x00+x10)y11
    # M6 = (x01-x00)(y00+y10)
    # M7 = (x11-x01)(y01+y11)
    
    # U (x part):
    # col 0: x00+x11 -> [1, 0, 0, 1] (i*2+j: 00=0, 01=1, 10=2, 11=3)
    # col 1: x10+x11 -> [0, 0, 1, 1]
    # col 2: x00 -> [1, 0, 0, 0]
    # col 3: x11 -> [0, 0, 0, 1]
    # col 4: x00+x10 -> [1, 0, 1, 0]
    # col 5: x01-x00 -> [-1, 1, 0, 0]
    # col 6: x11-x01 -> [0, -1, 0, 1]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (idx 0)
        [0, 0, 0, 0, 0, 1, -1], # i=0, j=1 (idx 1)
        [0, 1, 0, 0, 1, 0, 0], # i=1, j=0 (idx 2)
        [1, 1, 0, 1, 0, 0, 1]  # i=1, j=1 (idx 3)
    ]
    
    # V (y part):
    # col 0: y00+y11 -> [1, 0, 0, 1] (j*2+k: 00=0, 01=1, 10=2, 11=3)
    # col 1: y00 -> [1, 0, 0, 0]
    # col 2: y10-y11 -> [0, 0, 1, -1]
    # col 3: y01+y11 -> [0, 1, 0, 1]
    # col 4: y11 -> [0, 0, 0, 1]
    # col 5: y00+y10 -> [1, 0, 1, 0]
    # col 6: y01+y11 -> [0, 1, 0, 1]
    
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0], # j=0, k=0 (idx 0)
        [0, 0, 0, 1, 0, 0, 1], # j=0, k=1 (idx 1)
        [0, 0, 1, 0, 0, 1, 0], # j=1, k=0 (idx 2)
        [1, 0, -1, 1, 1, 0, 1] # j=1, k=1 (idx 3)
    ]
    
    # W (z part):
    # z00 = M1 + M4 - M5 - M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # W[k*2+i, r]
    # idx 0: k=0, i=0 (z00)
    # idx 1: k=0, i=1 (z10)
    # idx 2: k=1, i=0 (z01)
    # idx 3: k=1, i=1 (z11)
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, -1], # z00
        [0, 1, 0, 1, 0, 0, 0],   # z10
        [0, 0, 1, 0, 1, 0, 0],   # z01
        [1, -1, 1, 0, 0, 1, 0]   # z11
    ]
    
    current_col = 0
    # Iterate over the 3 blocks of B1
    for block_idx in range(3):
        # Block covers k in {2*block_idx, 2*block_idx+1}
        k_offset = 2 * block_idx
        
        for r in range(7):
            # Map Strassen indices to global indices
            # U: i in 0..1, j in 0..1 (m1=2)
            for i in range(2):
                for j in range(2):
                    u_val = U_strassen[i * 2 + j][r]
                    if u_val != 0:
                        U[i * m + j][current_col] = u_val
            
            # V: j in 0..1 (m1=2), k in 0..1 (local) -> global k = k_offset + k_local
            for j in range(2):
                for k_local in range(2):
                    v_val = V_strassen[j * 2 + k_local][r]
                    if v_val != 0:
                        k_global = k_offset + k_local
                        V[j * p + k_global][current_col] = v_val
            
            # W: k in 0..1 (local) -> global k, i in 0..1
            for k_local in range(2):
                for i in range(2):
                    w_val = W_strassen[k_local * 2 + i][r]
                    if w_val != 0:
                        k_global = k_offset + k_local
                        W[k_global * n + i][current_col] = w_val
            
            current_col += 1
    
    # --- Part 2: Schoolbook for 2x3 by 3x6 ---
    # A2 is 2x3, B2 is 3x6
    # i in 0..1
    # j in 2..4 (global j)
    # k in 0..5
    
    for i in range(2):
        for j_local in range(3):
            j_global = m1 + j_local # j starts at 2
            for k in range(6):
                U[i * m + j_global][current_col] = 1
                V[j_global * p + k][current_col] = 1
                W[k * n + i][current_col] = 1
                current_col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
