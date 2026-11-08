import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 2 j-blocks
    # Each j-block has 3 k-pairs (Strassen) and 1 k-single (Schoolbook)
    # Strassen rank: 7
    # Schoolbook rank: 4
    rank_per_j_block = 3 * 7 + 1 * 4
    total_rank = 2 * rank_per_j_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for s in range(2): # j block
        # k blocks
        for t in range(3): # Strassen blocks
            # Get local factors for M_2,2,2
            U_loc, V_loc, W_loc = get_strassen_222()
            
            # Map to global
            for r in range(7):
                # U
                for i in range(2):
                    for jl in range(2):
                        jg = 2*s + jl
                        ug_idx = i * m + jg
                        U[ug_idx][col_offset + r] += U_loc[i * 2 + jl][r]
                        
                        # V
                        kg = 2*t + (r % 2) # This mapping is wrong, V_loc indices are fixed
                        # We need to iterate over V_loc indices properly
                        # V_loc is 4x7. Indices (j, k) local.
                        # We need to add V_loc[jl * 2 + kl][r] to V[(2s+jl)*p + (2t+kl)][col_offset+r]
                        # Let's do it in the loop below instead of inside r loop if possible,
                        # or just iterate all local indices.
            
            # Better approach: iterate local indices and add to global.
            
            for r in range(7):
                # U
                for i in range(2):
                    for jl in range(2):
                        jg = 2*s + jl
                        ug_idx = i * m + jg
                        U[ug_idx][col_offset + r] += U_loc[i * 2 + jl][r]
                        
                # V
                for jl in range(2):
                    for kl in range(2):
                        jg = 2*s + jl
                        kg = 2*t + kl
                        vg_idx = jg * p + kg
                        V[vg_idx][col_offset + r] += V_loc[jl * 2 + kl][r]
                        
                # W
                for i in range(2):
                    for kl in range(2):
                        kg = 2*t + kl
                        wg_idx = kg * n + i
                        W[wg_idx][col_offset + r] += W_loc[kl * 2 + i][r] # W_loc index for C is k*n+i?
                        # Local C is 2x2. Index k*n+i?
                        # Task W is p*n. Local W is 2*2=4.
                        # Local C index (i, k). Flat k*2+i.
                        # W_loc[kl * 2 + i][r].
                        
            col_offset += 7
            
        # t = 3 (Schoolbook)
        U_loc, V_loc, W_loc = get_schoolbook_221()
        
        for r in range(4):
            # U
            for i in range(2):
                for jl in range(2):
                    jg = 2*s + jl
                    ug_idx = i * m + jg
                    U[ug_idx][col_offset + r] += U_loc[i * 2 + jl][r]
                    
            # V
            for jl in range(2):
                kg = 6 # single k
                vg_idx = (2*s + jl) * p + kg
                V[vg_idx][col_offset + r] += V_loc[jl * 1 + 0][r] # V_loc is 2x4?
                # M_2,2,1. B is 2x1. V size 2*1=2.
                # V_loc[jl * 1 + 0][r].
                
            # W
            for i in range(2):
                kg = 6
                wg_idx = kg * n + i
                W[wg_idx][col_offset + r] += W_loc[0 * 2 + i][r] # W_loc is 2x4?
                # C is 2x1. W size 1*2=2.
                # W_loc[0 * 2 + i][r].
                
        col_offset += 4

    return {"U": U, "V": V, "W": W}

def get_strassen_222():
    # Returns U, V, W for M_2,2,2
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1 # C00, C01, C10, C11?
    # C00 = M1 + ...
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So W for M1 contributes to C00 and C11.
    # C00 index 0. C11 index 3.
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1 # C10
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1 # C01, C11
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1 # C00, C10
    
    # M5: (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[1][4] += 1; W[0][4] -= 1 # C01, C00
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] -= 1 # C00, C11
    
    return U, V, W

def get_schoolbook_221():
    # Returns U, V, W for M_2,2,1
    # U: 4x4, V: 2x4, W: 2x4
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Terms:
    # 1: A00 B0 -> C0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # 2: A01 B1 -> C0
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # 3: A10 B0 -> C1
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # 4: A11 B1 -> C1
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W
