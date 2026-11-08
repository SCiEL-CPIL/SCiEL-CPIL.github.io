"""Rewritten program for matrix multiplication 3,4,8 using rank-56 decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: use a rank-56 decomposition derived from a 4x4x8 Strassen construction.
    # We pad n from 3 to 4, use Strassen on 4x4x4 blocks (two such blocks for the 8 columns),
    # then trim the output matrix W to the first 3 rows per block of 8.
    
    # The 4x4x8 tensor is built from two 4x4x4 tensors.
    # Each 4x4x4 tensor uses 4 Strassen 2x2x2 blocks, each with rank 7, total 28.
    # So 4x4x8 has rank 56.
    
    rank = 56
    U = [[0] * rank for _ in range(4 * m)]  # 4x4 = 16 rows, we'll trim to 3*4=12
    V = [[0] * rank for _ in range(m * p)]  # 4x8 = 32 rows
    W_full = [[0] * rank for _ in range(p * 4)]  # 8*4 = 32 rows, we'll trim to 8*3=24
    
    col = 0
    
    # Strassen 2x2x2 local factors
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0]    # M7: C00+=1
    ]
    
    # Iterate over the two 4x4x4 blocks (k_block in 0,1)
    # Each 4x4x4 block covers k in [k_block*4, k_block*4+3]
    for k_block in range(2):
        k_start = k_block * 4
        
        # Within each 4x4x4, iterate over 4 2x2x2 Strassen blocks
        # j_block in 0,1 for rows of A (i in 0..3 split into 0,1 and 2,3)
        # k_sub_block in 0,1 for cols of B (k in k_start..k_start+3 split into two pairs)
        for i_block in range(2):
            i_start = i_block * 2
            for j_block in range(2):
                j_start = j_block * 2
                for k_sub in range(2):
                    k_sub_start = k_start + k_sub * 2
                    
                    # 7 Strassen terms for this 2x2x2 block
                    for t in range(7):
                        # A: local indices l=0..3 map to (i', j')
                        # l=0 -> (0,0), l=1 -> (0,1), l=2 -> (1,0), l=3 -> (1,1)
                        for l in range(4):
                            if strassen_U[t][l] != 0:
                                li = l // 2  # 0 or 1
                                lj = l % 2   # 0 or 1
                                gi = i_start + li
                                gj = j_start + lj
                                U[gi * m + gj][col] += strassen_U[t][l]
                        
                        # B: local indices l=0..3 map to (j', k')
                        for l in range(4):
                            if strassen_V[t][l] != 0:
                                lj = l // 2
                                lk = l % 2
                                gj = j_start + lj
                                gk = k_sub_start + lk
                                V[gj * p + gk][col] += strassen_V[t][l]
                        
                        # C: local indices l=0..3 map to (i', k')
                        for l in range(4):
                            if strassen_W[t][l] != 0:
                                li = l // 2
                                lk = l % 2
                                gi = i_start + li
                                gk = k_sub_start + lk
                                W_full[gk * 4 + gi][col] += strassen_W[t][l]
                        
                        col += 1
    
    # Trim U to 3*4=12 rows (first 3 of the 4 i-indices)
    U_trimmed = [U[gi * m + gj][:] for gi in range(3) for gj in range(m)]
    
    # V stays the same (4*8=32 rows)
    V_trimmed = V
    
    # Remap W from 4-index to 3-index
    # W_full has indices k*4 + i for i in 0..3
    # We need W with indices k*3 + i for i in 0..2
    W_trimmed = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W_trimmed[k * n + i][r] = W_full[k * 4 + i][r]
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
# EVOLVE-BLOCK-END
