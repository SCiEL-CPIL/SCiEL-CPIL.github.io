import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # We will construct a decomposition of rank 46.
    # 3 blocks of 4x2 (rank 14 each) and 1 block of 4x1 (rank 4).
    # Total rank = 3 * 14 + 4 = 46.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)] # 8 x 46
    V = [[0] * rank for _ in range(m * p)] # 28 x 46
    W = [[0] * rank for _ in range(p * n)] # 14 x 46
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # U indices: i*m + j. For 2x2, i in 0..1, j in 0..1.
    # But in our 2x4 A, the 2x2 blocks are A1 (cols 0,1) and A2 (cols 2,3).
    # For B 4x2, the 2x2 blocks are B_top (rows 0,1) and B_bot (rows 2,3).
    
    def add_strassen_block(a_col_offset, b_row_offset, c_col_offset):
        nonlocal col
        # A1 is at a_col_offset (0 or 2)
        # B_top is at b_row_offset (0)
        # B_bot is at b_row_offset + 2 (2)
        
        # Strassen for A1 * B_top
        # A1 indices: i in 0..1, j in a_col_offset..a_col_offset+1
        # B_top indices: j in 0..1, k in c_col_offset..c_col_offset+1
        
        # M1 = (a11 + a22) * (b11 + b22)
        # U: a11, a22. V: b11, b22. W: c11, c22 (with signs)
        
        # This is complex to map directly.
        # Let's use a simpler approach:
        # Just use schoolbook for 2x2x2? Rank 4.
        # Then R(2,4,2) = 8.
        # R(2,4,7) = 3*8 + 4 = 28.
        # This is much better!
        
        # Schoolbook for 2x2x2:
        # c11 = a11 b11 + a12 b21
        # c12 = a11 b12 + a12 b22
        # c21 = a21 b11 + a22 b21
        # c22 = a21 b12 + a22 b22
        
        # 4 terms.
        # 1. a11 b11 -> c11
        U[a_col_offset + 0][col] = 1 # a11 (i=0, j=a_col_offset)
        V[b_row_offset + 0][col] = 1 # b11 (j=0, k=c_col_offset)
        W[c_col_offset * n + 0][col] = 1 # c11 (k=c_col_offset, i=0)
        col += 1
        
        # 2. a12 b21 -> c11
        U[a_col_offset + 1][col] = 1 # a12 (i=0, j=a_col_offset+1)
        V[b_row_offset + 2][col] = 1 # b21 (j=2, k=c_col_offset)
        W[c_col_offset * n + 0][col] = 1 # c11
        col += 1
        
        # 3. a11 b12 -> c12
        U[a_col_offset + 0][col] = 1 # a11
        V[b_row_offset + 1][col] = 1 # b12 (j=0, k=c_col_offset+1)
        W[(c_col_offset+1) * n + 0][col] = 1 # c12
        col += 1
        
        # 4. a12 b22 -> c12
        U[a_col_offset + 1][col] = 1 # a12
        V[b_row_offset + 3][col] = 1 # b22 (j=2, k=c_col_offset+1)
        W[(c_col_offset+1) * n + 0][col] = 1 # c12
        col += 1
        
        # 5. a21 b11 -> c21
        U[a_col_offset + 2][col] = 1 # a21 (i=1, j=a_col_offset)
        V[b_row_offset + 0][col] = 1 # b11
        W[c_col_offset * n + 1][col] = 1 # c21
        col += 1
        
        # 6. a22 b21 -> c21
        U[a_col_offset + 3][col] = 1 # a22 (i=1, j=a_col_offset+1)
        V[b_row_offset + 2][col] = 1 # b21
        W[c_col_offset * n + 1][col] = 1 # c21
        col += 1
        
        # 7. a21 b12 -> c22
        U[a_col_offset + 2][col] = 1 # a21
        V[b_row_offset + 1][col] = 1 # b12
        W[(c_col_offset+1) * n + 1][col] = 1 # c22
        col += 1
        
        # 8. a22 b22 -> c22
        U[a_col_offset + 3][col] = 1 # a22
        V[b_row_offset + 3][col] = 1 # b22
        W[(c_col_offset+1) * n + 1][col] = 1 # c22
        col += 1

    # Wait, the above is for 2x2x2 schoolbook.
    # But we have A 2x4 and B 4x2.
    # A = [A1, A2]. B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # So we need to add Strassen/Schoolbook for A1 B1 and A2 B2.
    
    # Let's restart the block addition.
    col = 0
    
    def add_block_2x4x2(b_col_offset):
        # b_col_offset is the column index in B (0, 2, 4)
        # A is 2x4. B is 4x2.
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        # B1 = B[0:2, :], B2 = B[2:4, :]
        # C1 = A1 B1, C2 = A2 B2.
        
        # Schoolbook for A1 B1 (2x2x2)
        # A1 indices: i in 0..1, j in 0..1
        # B1 indices: j in 0..1, k in b_col_offset..b_col_offset+1
        
        # 1. a11 b11 -> c11
        U[0][col] = 1
        V[b_col_offset][col] = 1
        W[b_col_offset * 2 + 0][col] = 1
        col += 1
        
        # 2. a12 b21 -> c11
        U[1][col] = 1
        V[2 + b_col_offset][col] = 1
        W[b_col_offset * 2 + 0][col] = 1
        col += 1
        
        # 3. a11 b12 -> c12
        U[0][col] = 1
        V[1 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 0][col] = 1
        col += 1
        
        # 4. a12 b22 -> c12
        U[1][col] = 1
        V[3 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 0][col] = 1
        col += 1
        
        # 5. a21 b11 -> c21
        U[2][col] = 1
        V[b_col_offset][col] = 1
        W[b_col_offset * 2 + 1][col] = 1
        col += 1
        
        # 6. a22 b21 -> c21
        U[3][col] = 1
        V[2 + b_col_offset][col] = 1
        W[b_col_offset * 2 + 1][col] = 1
        col += 1
        
        # 7. a21 b12 -> c22
        U[2][col] = 1
        V[1 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 1][col] = 1
        col += 1
        
        # 8. a22 b22 -> c22
        U[3][col] = 1
        V[3 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 1][col] = 1
        col += 1
        
        # Now A2 B2
        # A2 indices: i in 0..1, j in 2..3
        # B2 indices: j in 2..3, k in b_col_offset..b_col_offset+1
        
        # 9. a11 b11 -> c11
        U[4][col] = 1 # A2[0,0] is A[0,2]
        V[2 + b_col_offset][col] = 1 # B2[0,0] is B[2, b_col_offset]
        W[b_col_offset * 2 + 0][col] = 1
        col += 1
        
        # 10. a12 b21 -> c11
        U[5][col] = 1
        V[4 + b_col_offset][col] = 1
        W[b_col_offset * 2 + 0][col] = 1
        col += 1
        
        # 11. a11 b12 -> c12
        U[4][col] = 1
        V[3 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 0][col] = 1
        col += 1
        
        # 12. a12 b22 -> c12
        U[5][col] = 1
        V[5 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 0][col] = 1
        col += 1
        
        # 13. a21 b11 -> c21
        U[6][col] = 1
        V[2 + b_col_offset][col] = 1
        W[b_col_offset * 2 + 1][col] = 1
        col += 1
        
        # 14. a22 b21 -> c21
        U[7][col] = 1
        V[4 + b_col_offset][col] = 1
        W[b_col_offset * 2 + 1][col] = 1
        col += 1
        
        # 15. a21 b12 -> c22
        U[6][col] = 1
        V[3 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 1][col] = 1
        col += 1
        
        # 16. a22 b22 -> c22
        U[7][col] = 1
        V[5 + b_col_offset][col] = 1
        W[(b_col_offset+1) * 2 + 1][col] = 1
        col += 1

    # Add 3 blocks of 4x2
    add_block_2x4x2(0)
    add_block_2x4x2(2)
    add_block_2x4x2(4)
    
    # Add 1 block of 4x1
    # A is 2x4. B is 4x1.
    # C is 2x1.
    # c11 = sum a1j b_j1
    # c21 = sum a2j b_j1
    
    # 1. a11 b11
    U[0][col] = 1
    V[6][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # 2. a12 b21
    U[1][col] = 1
    V[7][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # 3. a13 b31
    U[2][col] = 1
    V[8][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # 4. a14 b41
    U[3][col] = 1
    V[9][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # 5. a21 b11
    U[4][col] = 1
    V[6][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    # 6. a22 b21
    U[5][col] = 1
    V[7][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    # 7. a23 b31
    U[6][col] = 1
    V[8][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    # 8. a24 b41
    U[7][col] = 1
    V[9][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    # Total rank used: 3 * 16 + 8 = 56.
    # This is just schoolbook!
    # I made a mistake. Schoolbook for 2x2x2 is 4, not 8.
    # I listed 8 terms for 2x2x2.
    # 2x2x2 has 4 products.
    # I duplicated them.
    
    # Let's correct.
    # 2x2x2 schoolbook:
    # c11 = a11 b11 + a12 b21
    # c12 = a11 b12 + a12 b22
    # c21 = a21 b11 + a22 b21
    # c22 = a21 b12 + a22 b22
    
    # So 4 terms per 2x2x2.
    # R(2,4,2) = 8.
    # R(2,4,7) = 3 * 8 + 8 = 32.
    
    # Let's re-implement with rank 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_block_2x4x2(b_col_offset):
        nonlocal col
        # A1 B1
        U[0][col] = 1; V[b_col_offset][col] = 1; W[b_col_offset * 2 + 0][col] = 1; col += 1
        U[1][col] = 1; V[2 + b_col_offset][col] = 1; W[b_col_offset * 2 + 0][col] = 1; col += 1
        U[0][col] = 1; V[1 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 0][col] = 1; col += 1
        U[1][col] = 1; V[3 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 0][col] = 1; col += 1
        U[2][col] = 1; V[b_col_offset][col] = 1; W[b_col_offset * 2 + 1][col] = 1; col += 1
        U[3][col] = 1; V[2 + b_col_offset][col] = 1; W[b_col_offset * 2 + 1][col] = 1; col += 1
        U[2][col] = 1; V[1 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 1][col] = 1; col += 1
        U[3][col] = 1; V[3 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 1][col] = 1; col += 1
        
        # A2 B2
        U[4][col] = 1; V[2 + b_col_offset][col] = 1; W[b_col_offset * 2 + 0][col] = 1; col += 1
        U[5][col] = 1; V[4 + b_col_offset][col] = 1; W[b_col_offset * 2 + 0][col] = 1; col += 1
        U[4][col] = 1; V[3 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 0][col] = 1; col += 1
        U[5][col] = 1; V[5 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 0][col] = 1; col += 1
        U[6][col] = 1; V[2 + b_col_offset][col] = 1; W[b_col_offset * 2 + 1][col] = 1; col += 1
        U[7][col] = 1; V[4 + b_col_offset][col] = 1; W[b_col_offset * 2 + 1][col] = 1; col += 1
        U[6][col] = 1; V[3 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 1][col] = 1; col += 1
        U[7][col] = 1; V[5 + b_col_offset][col] = 1; W[(b_col_offset+1) * 2 + 1][col] = 1; col += 1

    add_block_2x4x2(0)
    add_block_2x4x2(2)
    add_block_2x4x2(4)
    
    # Block 4x1
    U[0][col] = 1; V[6][col] = 1; W[6 * 2 + 0][col] = 1; col += 1
    U[1][col] = 1; V[7][col] = 1; W[6 * 2 + 0][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6 * 2 + 0][col] = 1; col += 1
    U[3][col] = 1; V[9][col] = 1; W[6 * 2 + 0][col] = 1; col += 1
    U[4][col] = 1; V[6][col] = 1; W[6 * 2 + 1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[6 * 2 + 1][col] = 1; col += 1
    U[6][col] = 1; V[8][col] = 1; W[6 * 2 + 1][col] = 1; col += 1
    U[7][col] = 1; V[9][col] = 1; W[6 * 2 + 1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
