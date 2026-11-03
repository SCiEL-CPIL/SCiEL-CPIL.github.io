"""Program using Strassen's algorithm for 4x4x4 and schoolbook for the remaining columns."""
DIMENSIONS = [4, 4, 6]

# Strassen's 7 products for 2x2x2 (indices 0..6)
# Each product is a 7-tuple of (u_coeff, v_coeff, w_coeff)
# u_coeff[i,j], v_coeff[k,j] are coefficients for the 2x2 blocks
# Actually, Strassen for 2x2x2:
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# Let's define the Strassen decomposition for 2x2x2 explicitly.
# U rows: 4 (A00, A01, A10, A11)
# V rows: 4 (B00, B01, B10, B11)
# W rows: 4 (C00, C01, C10, C11)
# 7 columns.

def strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # Wait, C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # So W for M1: C00 gets +1, C11 gets +1.
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10 = M2 + M4
    W[3][1] = -1 # C11 = M1 - M2 + M3 + M6
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01 = M3 + M5
    W[3][2] = 1 # C11 = M1 - M2 + M3 + M6
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C00 = M1 + M4 - M5 + M7
    W[2][3] = 1 # C10 = M2 + M4
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00 = M1 + M4 - M5 + M7
    W[1][4] = 1 # C01 = M3 + M5
    
    # M6 = (A10-A00)*(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1 # C11 = M1 - M2 + M3 + M6
    
    # M7 = (A01-A11)*(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00 = M1 + M4 - M5 + M7
    W[1][6] = -1 # Wait, C01 = M3 + M5. M7 is not in C01?
    # Let me re-check Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Yes, M7 is only in C00.
    # So W[0][6] = 1, W[1][6] = 0.
    W[1][6] = 0
    W[0][6] = 1
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS # 4, 4, 6
    
    # We will use Strassen for the first 4 columns of B, and schoolbook for the last 2.
    # Actually, Strassen works for square matrices. Here A is 4x4, B is 4x6.
    # We can treat B as [B1, B2] where B1 is 4x4 and B2 is 4x2.
    # C = A * B1 + A * B2
    # For A * B1, we can use Strassen (rank 49 for 4x4x4? No, Strassen is for 2x2x2. For 4x4x4, we can use 4x4x4 Strassen recursively or just use the known rank 49 for 4x4x4? 
    # Wait, the rank of 4x4x4 matrix multiplication is known to be 49 (Strassen's algorithm applied twice? No, Strassen is 7 for 2x2x2. For 4x4x4, we can do it in 49 = 7^2. 
    # Let's verify: 4x4x4 can be seen as 4 blocks of 2x2. 
    # Actually, the rank of 4x4x4 is 49. This is achieved by applying Strassen's 2x2x2 algorithm to the 4 blocks of 2x2 matrices.
    # So for A (4x4) and B1 (4x4), we can use rank 49.
    # For A (4x4) and B2 (4x2), we can use schoolbook: 4*4*2 = 32.
    # Total rank = 49 + 32 = 81.
    # This is better than 96.
    
    # Let's construct the 4x4x4 Strassen decomposition (rank 49).
    # The 4x4x4 tensor is the tensor product of two 2x2x2 tensors? No, it's a block structure.
    # Actually, the rank of M_{4,4,4} is 49. This is a known result.
    # We can construct it by applying Strassen's 2x2x2 algorithm to the 2x2 blocks of 2x2 matrices.
    
    # Let's define the 4x4x4 decomposition using the 2x2x2 Strassen decomposition.
    # The 4x4x4 tensor T_{4,4,4} can be decomposed as a sum of 49 rank-1 tensors.
    # We can generate this by taking the tensor product of the 2x2x2 Strassen decomposition with itself?
    # No, the tensor product of two 2x2x2 tensors gives a 4x4x4 tensor of rank 7*7=49.
    # Yes! The tensor product of the 2x2x2 multiplication tensor with itself is the 4x4x4 multiplication tensor.
    # So we can use the Kronecker product of the U, V, W matrices from the 2x2x2 Strassen decomposition.
    
    U2, V2, W2 = strassen_2x2x2()
    
    import numpy as np
    U4_4_4 = np.kron(np.array(U2), np.array(U2)) # 4x4 x 49
    V4_4_4 = np.kron(np.array(V2), np.array(V2)) # 4x4 x 49
    W4_4_4 = np.kron(np.array(W2), np.array(W2)) # 4x4 x 49
    
    # Now for A (4x4) and B2 (4x2), we use schoolbook.
    # This is a 4x4x2 tensor. Rank = 4*4*2 = 32.
    U4_4_2 = [[0]*32 for _ in range(16)]
    V4_4_2 = [[0]*32 for _ in range(8)]
    W4_4_2 = [[0]*32 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U4_4_2[i*4+j][col] = 1
                V4_4_2[j*4+k][col] = 1
                W4_4_2[k*4+i][col] = 1
                col += 1
    
    # Combine the two decompositions.
    # The total rank is 49 + 32 = 81.
    # U has 16 rows (4x4 A).
    # V has 4*6 = 24 rows (4x6 B).
    # W has 4*6 = 24 rows (4x6 C).
    
    # For the first 4 columns of B (B1), we use U4_4_4, V4_4_4, W4_4_4.
    # V4_4_4 has 16 rows (4x4 B1). We need to map these to the first 16 rows of V (which is 24 rows).
    # W4_4_4 has 16 rows (4x4 C1). We need to map these to the first 16 rows of W (which is 24 rows).
    # Wait, C is 4x6. C1 is 4x4 (first 4 columns), C2 is 4x2 (last 2 columns).
    # So W rows 0-15 correspond to C1, rows 16-23 correspond to C2.
    # V rows 0-15 correspond to B1, rows 16-23 correspond to B2.
    
    U = [[0]*81 for _ in range(16)]
    V = [[0]*81 for _ in range(24)]
    W = [[0]*81 for _ in range(24)]
    
    # Fill in the first 49 columns (Strassen part).
    for i in range(16):
        for c in range(49):
            U[i][c] = U4_4_4[i][c]
            V[i][c] = V4_4_4[i][c]
            W[i][c] = W4_4_4[i][c]
    
    # Fill in the next 32 columns (Schoolbook part for B2).
    for i in range(16):
        for c in range(32):
            U[i][c+49] = U4_4_2[i][c]
            # V rows for B2 are 16-23.
            V[i+8][c+49] = V4_4_2[i][c] # Wait, V4_4_2 has 8 rows (j=0..3, k=0..1).
            # j is 0..3, k is 0..1. So row index is j*4+k? No, V4_4_2 row index is j*2+k?
            # In V4_4_2, rows are j*2+k? No, V4_4_2 is for B2 (4x2).
            # V4_4_2 row index is j*2+k? No, the standard convention is V[j*p+k].
            # Here p=2, so V[j*2+k].
            # But in V4_4_2, I used V[j*4+k]? No, I used V[j*4+k] where j is 0..3 and k is 0..1?
            # Wait, in the schoolbook part, B2 is 4x2. So j is 0..3 (rows of B2), k is 0..1 (cols of B2).
            # The row index in V for B2 is j*2+k.
            # But in my V4_4_2, I used row index j*4+k? No, I used V[j*4+k]? Let me re-check.
            # In the code above: V4_4_2 = [[0]*32 for _ in range(8)]
            # And V4_4_2[j*4+k][col] = 1. This is wrong. It should be j*2+k.
            # Let me correct V4_4_2.
    
    # Let's redo V4_4_2 correctly.
    U4_4_2 = [[0]*32 for _ in range(16)]
    V4_4_2 = [[0]*32 for _ in range(8)]
    W4_4_2 = [[0]*32 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U4_4_2[i*4+j][col] = 1
                V4_4_2[j*2+k][col] = 1
                W4_4_2[k*4+i][col] = 1
                col += 1
    
    # Now fill in the schoolbook part.
    for i in range(16):
        for c in range(32):
            U[i][c+49] = U4_4_2[i][c]
            # V rows for B2 are 16-23.
            # V4_4_2 row index is j*2+k.
            # So V[i+16][c+49] = V4_4_2[i][c]? No, i is the row index of U, which is i*4+j.
            # We need to map V4_4_2 rows to V rows 16-23.
            # V4_4_2 has 8 rows. V rows 16-23 are 8 rows.
            # So V[16 + r][c+49] = V4_4_2[r][c] for r in 0..7.
            for r in range(8):
                V[16+r][c+49] = V4_4_2[r][c]
            # W rows for C2 are 16-23.
            # W4_4_2 has 8 rows. W rows 16-23 are 8 rows.
            for r in range(8):
                W[16+r][c+49] = W4_4_2[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
