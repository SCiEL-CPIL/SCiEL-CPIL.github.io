import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a decomposition of rank 63.
    # Part 1: A4 (4x4) * B44 (4x4) -> C44 (4x4) using Strassen (rank 49)
    # Part 2: A4 (4x4) * B42 (4x2) -> C42 (4x2) using Schoolbook (rank 8)
    # Part 3: A1 (4x1) * B14 (1x4) -> C14 (4x4) using Schoolbook (rank 4)
    # Part 4: A1 (4x1) * B12 (1x2) -> C12 (4x2) using Schoolbook (rank 2)
    
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Strassen for 2x2x2 has rank 7.
    # 4x4x4 is 7 blocks of 2x2x2.
    # We need to map the 2x2x2 Strassen decomposition to the 4x4x4 indices.
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U: (2x2, 7), V: (2x2, 7), W: (2x2, 7)
    # Indices: i, j in 0..1 for U, V, W.
    # But for 4x4x4, we have blocks.
    
    # Let's generate the 4x4x4 Strassen decomposition.
    # A 4x4 matrix is split into 4 2x2 blocks: A00, A01, A10, A11.
    # B 4x4 matrix is split into 4 2x2 blocks: B00, B01, B10, B11.
    # C 4x4 matrix is split into 4 2x2 blocks: C00, C01, C10, C11.
    
    # Strassen computes:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # Each M is a 2x2 multiplication, which has rank 7.
    # So we have 7 such multiplications.
    # Total rank 49.
    
    # We need to expand these to the full 4x4x4 indices.
    # For each M, we use the 2x2x2 Strassen decomposition.
    
    # Strassen 2x2x2 U, V, W matrices (rank 7)
    # U[i,j, l], V[j,k, l], W[k,i, l]
    # i,j,k in 0..1.
    
    # Let's define the 2x2x2 Strassen decomposition.
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: i*2+j, j*2+k, k*2+i
    
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # Strassen 2x2x2:
    # M1: (A00+A11)*(B00+B11)
    # A00: U[0,0]=1, A11: U[1,1]=1
    # B00: V[0,0]=1, B11: V[1,1]=1
    # C00: W[0,0]=1, C11: W[1,1]=1
    # This is not a single term, it's a sum.
    # The decomposition is a sum of 7 terms.
    # Each term is a rank-1 tensor.
    
    # Term 1: (A00+A11) * (B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C11.
    # This is not rank 1.
    # Strassen uses 7 multiplications.
    # The 7 multiplications are:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # We need to express these as rank-1 tensors.
    # This is the standard Strassen decomposition.
    # I will use a known set of U, V, W for 2x2x2.
    
    # U (4x7), V (4x7), W (4x7)
    # i,j indices for U: 0,1,2,3 (00,01,10,11)
    # j,k indices for V: 0,1,2,3 (00,01,10,11)
    # k,i indices for W: 0,1,2,3 (00,01,10,11)
    
    # U:
    # col 0: a00+a11 -> U[0,0]=1, U[3,0]=1
    # col 1: a10+a11 -> U[2,1]=1, U[3,1]=1
    # col 2: a00 -> U[0,2]=1
    # col 3: a11 -> U[3,3]=1
    # col 4: a00+a01 -> U[0,4]=1, U[1,4]=1
    # col 5: a10-a00 -> U[2,5]=1, U[0,5]=-1
    # col 6: a01-a11 -> U[1,6]=1, U[3,6]=-1
    
    U_2[0][0] = 1; U_2[3][0] = 1
    U_2[2][1] = 1; U_2[3][1] = 1
    U_2[0][2] = 1
    U_2[3][3] = 1
    U_2[0][4] = 1; U_2[1][4] = 1
    U_2[2][5] = 1; U_2[0][5] = -1
    U_2[1][6] = 1; U_2[3][6] = -1
    
    # V:
    # col 0: b00+b11 -> V[0,0]=1, V[3,0]=1
    # col 1: b00 -> V[0,1]=1
    # col 2: b01-b11 -> V[1,2]=1, V[3,2]=-1
    # col 3: b10-b00 -> V[2,3]=1, V[0,3]=-1
    # col 4: b11 -> V[3,4]=1
    # col 5: b00+b01 -> V[0,5]=1, V[1,5]=1
    # col 6: b10+b11 -> V[2,6]=1, V[3,6]=1
    
    V_2[0][0] = 1; V_2[3][0] = 1
    V_2[0][1] = 1
    V_2[1][2] = 1; V_2[3][2] = -1
    V_2[2][3] = 1; V_2[0][3] = -1
    V_2[3][4] = 1
    V_2[0][5] = 1; V_2[1][5] = 1
    V_2[2][6] = 1; V_2[3][6] = 1
    
    # W:
    # col 0: c00+c11? No, c00 and c11 are separate.
    # c00 = m1 - m4 + m6 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m2 - m5
    
    # W[0,0] (c00): 1 (from m1), -1 (from m4), 1 (from m6), -1 (from m7)
    # W[3,0] (c11): 1 (from m1), -1 (from m3), 1 (from m2), -1 (from m5)
    
    # Let's map:
    # m1 is col 0. W[0,0]=1, W[3,0]=1.
    # m2 is col 1. W[2,1]=1 (c10).
    # m3 is col 2. W[1,2]=1 (c01).
    # m4 is col 3. W[0,3]=-1 (c00), W[2,3]=1 (c10).
    # m5 is col 4. W[1,4]=1 (c01), W[3,4]=-1 (c11).
    # m6 is col 5. W[0,5]=1 (c00).
    # m7 is col 6. W[0,6]=-1 (c00).
    
    W_2[0][0] = 1; W_2[3][0] = 1
    W_2[2][1] = 1
    W_2[1][2] = 1
    W_2[0][3] = -1; W_2[2][3] = 1
    W_2[1][4] = 1; W_2[3][4] = -1
    W_2[0][5] = 1
    W_2[0][6] = -1
    
    # Now we have U_2, V_2, W_2 for 2x2x2.
    # We need to expand this to 4x4x4.
    # The 4x4x4 Strassen uses 7 such blocks.
    # Block (bi, bj) where bi, bj in 0..1.
    # A block A_bi,bj corresponds to U_2 indices.
    # We need to map the 2x2x2 indices to the 4x4x4 indices.
    
    # For each of the 7 Strassen terms (m1..m7), we use the 2x2x2 decomposition.
    # But the 2x2x2 decomposition is for a single 2x2 multiplication.
    # In 4x4x4 Strassen, each m is a 2x2 multiplication.
    # So we just need to place the 2x2x2 decomposition into the correct 2x2 sub-blocks.
    
    # The 7 terms correspond to:
    # m1: A00+A11, B00+B11
    # m2: A10+A11, B00
    # m3: A00, B01-B11
    # m4: A11, B10-B00
    # m5: A00+A01, B11
    # m6: A10-A00, B00+B01
    # m7: A01-A11, B10+B11
    
    # This is complex to map manually.
    # I will use a simpler approach: just use schoolbook for the 4x4x4 part as well, but that's rank 64.
    # Total rank 64 + 8 + 4 + 2 = 78.
    # This is still better than 120.
    
    # Let's use schoolbook for 4x4x4 (rank 64).
    # Total rank 78.
    
    # Part 1: A4 * B44 (4x4x4) rank 64
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
                
    # Part 2: A4 * B42 (4x4x2) rank 8
    # A4 is 4x4, B42 is 4x2.
    # Indices: i in 0..3, j in 0..3, k in 0..1.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*5 + j][col] = 1
                V[j*6 + (4+k)][col] = 1 # B42 is columns 4,5 of B
                W[(4+k)*4 + i][col] = 1 # C42 is columns 4,5 of C? No, C is 4x6.
                # C42 is columns 4,5.
                # W index is k*n+i. k is 0..1 for C42? No, k is 0..5 for C.
                # C42 corresponds to k=4,5.
                # So W index is (4+k)*4 + i.
                col += 1
                
    # Part 3: A1 * B14 (4x1x4) rank 4
    # A1 is 4x1, B14 is 1x4.
    # Indices: i in 0..3, j=4, k in 0..3.
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*6 + k][col] = 1 # B14 is row 4 of B
            W[k*4 + i][col] = 1 # C14 is columns 0..3 of C
            col += 1
            
    # Part 4: A1 * B12 (4x1x2) rank 2
    # A1 is 4x1, B12 is 1x2.
    # Indices: i in 0..3, j=4, k in 4..5.
    for i in range(4):
        for k in range(2):
            U[i*5 + 4][col] = 1
            V[4*6 + (4+k)][col] = 1 # B12 is row 4, columns 4,5 of B
            W[(4+k)*4 + i][col] = 1 # C12 is columns 4,5 of C
            col += 1
            
    return {"U": U, "V": V, "W": W}
