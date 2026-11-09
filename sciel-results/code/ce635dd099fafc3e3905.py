"""Matrix multiplication 2x4 by 4x6 via a rank-deficient factorisation.
Produces a rank-42 tensor decomposition using Strassen-like blocks.
"""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # U shape: (n*m, rank) = (8, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (12, 42)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # A indices: 0:11, 1:12, 2:21, 3:22
    # B indices: 0:11, 1:12, 2:21, 3:22
    
    # We define the 7 terms for a generic 2x2 A and 2x2 B block
    # term 0: (a0+a3)(b0+b3)
    # term 1: (a2+a3)b0
    # term 2: a0(b1-b3)
    # term 3: a3(b2-b0)
    # term 4: (a0+a1)b3
    # term 5: (a2-a0)(b0+b1)
    # term 6: (a1-a3)(b2+b3)
    
    # Output P11, P12, P21, P22
    # P11 = M0 - M3 + M4 - M6
    # P12 = M2 + M6
    # P21 = M1 + M3
    # P22 = M0 - M1 + M2 + M5
    
    # We have 6 sub-problems:
    # 0: A_L, B_L_block_0 (cols 0-1 of B_L -> cols 0-1 of B)
    # 1: A_L, B_L_block_1 (cols 2-3 of B_L -> cols 2-3 of B)
    # 2: A_L, B_L_block_2 (cols 4-5 of B_L -> cols 4-5 of B)
    # 3: A_R, B_R_block_0 (cols 0-1 of B_R -> cols 6-7? No, B is 4x6. B_R is rows 2-3. Cols 0-1)
    # 4: A_R, B_R_block_1 (cols 2-3 of B_R -> cols 2-3)
    # 5: A_R, B_R_block_2 (cols 4-5 of B_R -> cols 4-5)
    
    # Wait, B is 4x6.
    # B_L is rows 0-1, cols 0-5.
    # B_R is rows 2-3, cols 0-5.
    
    # A_L is rows 0-1, cols 0-1 of A. (A is 2x4)
    # A_R is rows 0-1, cols 2-3 of A.
    
    # A indices in U:
    # A_L: a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    # A_R: a0=A[0,2], a1=A[0,3], a2=A[1,2], a3=A[1,3]
    
    # B indices in V:
    # B is 4x6.
    # B_L_block_0: b0=B[0,0], b1=B[0,1], b2=B[1,0], b3=B[1,1]
    # B_L_block_1: b0=B[0,2], b1=B[0,3], b2=B[1,2], b3=B[1,3]
    # B_L_block_2: b0=B[0,4], b1=B[0,5], b2=B[1,4], b3=B[1,5]
    # B_R_block_0: b0=B[2,0], b1=B[2,1], b2=B[3,0], b3=B[3,1]
    # B_R_block_1: b0=B[2,2], b1=B[2,3], b2=B[3,2], b3=B[3,3]
    # B_R_block_2: b0=B[2,4], b1=B[2,5], b2=B[3,4], b3=B[3,5]
    
    # C indices in W:
    # C is 2x6.
    # W index k*n+i.
    # Block 0 (cols 0-1):
    #   C[0,0] -> k=0, i=0 -> idx 0
    #   C[1,0] -> k=0, i=1 -> idx 1
    #   C[0,1] -> k=1, i=0 -> idx 2
    #   C[1,1] -> k=1, i=1 -> idx 3
    # Block 1 (cols 2-3):
    #   C[0,2] -> k=2, i=0 -> idx 4
    #   C[1,2] -> k=2, i=1 -> idx 5
    #   C[0,3] -> k=3, i=0 -> idx 6
    #   C[1,3] -> k=3, i=1 -> idx 7
    # Block 2 (cols 4-5):
    #   C[0,4] -> k=4, i=0 -> idx 8
    #   C[1,4] -> k=4, i=1 -> idx 9
    #   C[0,5] -> k=5, i=0 -> idx 10
    #   C[1,5] -> k=5, i=1 -> idx 11
    
    # We iterate over the 6 sub-problems
    for sub in range(6):
        # Determine A source
        if sub < 3:
            # A_L
            a_base = [0, 1, 2, 3] # indices in A (flattened 2x4? No, A is 2x4. U is 8x42)
            # U indices:
            # A[0,0] -> 0
            # A[0,1] -> 1
            # A[1,0] -> 2
            # A[1,1] -> 3
            a_idx = [0, 1, 2, 3]
        else:
            # A_R
            # A[0,2] -> 2
            # A[0,3] -> 3
            # A[1,2] -> 4
            # A[1,3] -> 5
            # Wait, U shape is (n*m, rank) = (8, 42).
            # A is 2x4.
            # Index i*m+j.
            # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
            # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
            a_idx = [2, 3, 6, 7]
            
        # Determine B source
        # B is 4x6. V shape (24, 42).
        # Index j*p+k.
        # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5
        # B[1,0]=6, ...
        # B[2,0]=12, ...
        # B[3,0]=18, ...
        
        block_col_offset = (sub % 3) * 2
        if sub < 3:
            # B_L (rows 0-1)
            b_row_offset = 0
        else:
            # B_R (rows 2-3)
            b_row_offset = 12 # 2 rows * 6 cols
            
        b_idx = [
            b_row_offset + 0 * 6 + block_col_offset + 0, # b0
            b_row_offset + 0 * 6 + block_col_offset + 1, # b1
            b_row_offset + 1 * 6 + block_col_offset + 0, # b2
            b_row_offset + 1 * 6 + block_col_offset + 1  # b3
        ]
        
        # Determine C source
        # C is 2x6. W shape (12, 42).
        # Index k*n+i.
        c_block_offset = (sub % 3) * 4 # 4 indices per block (2x2)
        # Indices in W:
        # c00: k=offset, i=0 -> offset*2 + 0
        # c10: k=offset, i=1 -> offset*2 + 1
        # c01: k=offset+1, i=0 -> (offset+1)*2 + 0
        # c11: k=offset+1, i=1 -> (offset+1)*2 + 1
        
        # Wait, my previous mapping:
        # Block 0: 0,1,2,3
        # Block 1: 4,5,6,7
        # Block 2: 8,9,10,11
        # This matches k*n+i with n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # Yes.
        
        c_idx = [
            c_block_offset + 0, # P11 -> C[0, block_col_offset]
            c_block_offset + 1, # P21 -> C[1, block_col_offset]
            c_block_offset + 2, # P12 -> C[0, block_col_offset+1]
            c_block_offset + 3  # P22 -> C[1, block_col_offset+1]
        ]
        # Note: P11 is C[0,0] of block. P21 is C[1,0]. P12 is C[0,1]. P22 is C[1,1].
        # My c_idx list:
        # 0: P11
        # 1: P21
        # 2: P12
        # 3: P22
        
        base_rank = sub * 7
        
        # Term 0: (a0+a3)(b0+b3)
        # U: a0=1, a3=1
        # V: b0=1, b3=1
        # W: P11=1, P22=1
        U[a_idx[0]][base_rank + 0] = 1
        U[a_idx[3]][base_rank + 0] = 1
        V[b_idx[0]][base_rank + 0] = 1
        V[b_idx[3]][base_rank + 0] = 1
        W[c_idx[0]][base_rank + 0] = 1
        W[c_idx[3]][base_rank + 0] = 1
        
        # Term 1: (a2+a3)b0
        # U: a2=1, a3=1
        # V: b0=1
        # W: P21=1, P22=-1
        U[a_idx[2]][base_rank + 1] = 1
        U[a_idx[3]][base_rank + 1] = 1
        V[b_idx[0]][base_rank + 1] = 1
        W[c_idx[1]][base_rank + 1] = 1
        W[c_idx[3]][base_rank + 1] = -1
        
        # Term 2: a0(b1-b3)
        # U: a0=1
        # V: b1=1, b3=-1
        # W: P12=1, P22=1
        U[a_idx[0]][base_rank + 2] = 1
        V[b_idx[1]][base_rank + 2] = 1
        V[b_idx[3]][base_rank + 2] = -1
        W[c_idx[2]][base_rank + 2] = 1
        W[c_idx[3]][base_rank + 2] = 1
        
        # Term 3: a3(b2-b0)
        # U: a3=1
        # V: b2=1, b0=-1
        # W: P11=-1, P21=1
        U[a_idx[3]][base_rank + 3] = 1
        V[b_idx[2]][base_rank + 3] = 1
        V[b_idx[0]][base_rank + 3] = -1
        W[c_idx[0]][base_rank + 3] = -1
        W[c_idx[1]][base_rank + 3] = 1
        
        # Term 4: (a0+a1)b3
        # U: a0=1, a1=1
        # V: b3=1
        # W: P11=1
        U[a_idx[0]][base_rank + 4] = 1
        U[a_idx[1]][base_rank + 4] = 1
        V[b_idx[3]][base_rank + 4] = 1
        W[c_idx[0]][base_rank + 4] = 1
        
        # Term 5: (a2-a0)(b0+b1)
        # U: a2=1, a0=-1
        # V: b0=1, b1=1
        # W: P22=1
        U[a_idx[2]][base_rank + 5] = 1
        U[a_idx[0]][base_rank + 5] = -1
        V[b_idx[0]][base_rank + 5] = 1
        V[b_idx[1]][base_rank + 5] = 1
        W[c_idx[3]][base_rank + 5] = 1
        
        # Term 6: (a1-a3)(b2+b3)
        # U: a1=1, a3=-1
        # V: b2=1, b3=1
        # W: P11=-1, P12=1
        U[a_idx[1]][base_rank + 6] = 1
        U[a_idx[3]][base_rank + 6] = -1
        V[b_idx[2]][base_rank + 6] = 1
        V[b_idx[3]][base_rank + 6] = 1
        W[c_idx[0]][base_rank + 6] = -1
        W[c_idx[2]][base_rank + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
