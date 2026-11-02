import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition
    # This uses a 2x2 Strassen block for the top-left 2x2 of A and B,
    # and handles the remaining terms with standard multiplications.
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (i, j), B (j, k), C (k, i)
    # A is 3x3, B is 3x3.
    # We partition A into A_00 (2x2), A_01 (2x1), A_10 (1x2), A_11 (1x1)
    # Similarly for B.
    
    # Strassen 2x2 for the top-left block
    # A_00 = [[A00, A01], [A10, A11]]
    # B_00 = [[B00, B10], [B01, B11]] (Note: B indices in V are j*p+k. B_00 is B[0,0], B[0,1], B[1,0], B[1,1])
    # Wait, V index is j*p+k.
    # B[0,0] is V[0*3+0]=V[0]
    # B[0,1] is V[0*3+1]=V[1]
    # B[1,0] is V[1*3+0]=V[3]
    # B[1,1] is V[1*3+1]=V[4]
    
    # U index is i*m+j.
    # A[0,0] is U[0]
    # A[0,1] is U[1]
    # A[1,0] is U[3]
    # A[1,1] is U[4]
    
    # W index is k*n+i.
    # C[0,0] is W[0]
    # C[0,1] is W[1]
    # C[1,0] is W[3]
    # C[1,1] is W[4]
    
    # Strassen terms for C_00 = A_00 B_00
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Assign columns 0-6 for Strassen
    # Col 0: M1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = -1; W[1][0] = 0; W[3][0] = 0 # Wait, W is for C.
    # C00 gets M1. C11 gets -M1.
    # So W[0][0] = 1 (C00), W[4][0] = -1 (C11).
    
    # Col 1: M2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1 # C10 gets M2
    
    # Col 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1 # C01 gets M3
    
    # Col 3: M4
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[3][3] = 1 # C00 gets M4, C10 gets M4
    
    # Col 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00 gets -M5, C01 gets M5
    
    # Col 5: M6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1 # C11 gets M6
    
    # Col 6: M7
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1 # C00 gets M7
    
    # Now handle the remaining terms involving row/col 2.
    # A has A20, A21, A22, A02, A12.
    # B has B20, B21, B22, B02, B12.
    # C has C20, C21, C22, C02, C12.
    
    # C22 = A20 B02 + A21 B12 + A22 B22
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C12 = A10 B02 + A11 B12 + A12 B22
    
    # We can reuse some Strassen terms? No, they are for A00..A11, B00..B11.
    # The new terms involve A2x, Ax2, B2x, Bx2.
    # We need 19 more multiplications to reach 26.
    # Actually, the cross terms are:
    # C20, C21, C22, C02, C12.
    # 5 elements.
    # Each is a sum of 3 products.
    # Total 15 products.
    # Plus C00..C11 are already done.
    # Wait, the Strassen part only covers C00, C01, C10, C11.
    # We still need to compute the contributions to C00..C11 from the 2nd row/col?
    # No, the partition was:
    # A = [[A22, a], [aT, alpha]]
    # The Strassen part computed A22 * B22.
    # But C00 = (A22 B22)00 + a0 b0.
    # So we need to add a0 b0 to C00.
    # Similarly for C01, C10, C11.
    # And compute C20, C21, C22, C02, C12 from scratch.
    
    # Terms needed:
    # a0 b0 (for C00)
    # a0 b1 (for C01)
    # a1 b0 (for C10)
    # a1 b1 (for C11)
    # a0 B20, a1 B20, alpha B20 (for C20) -> 3 terms
    # a0 B21, a1 B21, alpha B21 (for C21) -> 3 terms
    # a0 B22, a1 B22, alpha B22 (for C22) -> 3 terms
    # A20 b0, A21 b0, A22 b0 (for C02) -> 3 terms
    # A20 b1, A21 b1, A22 b1 (for C12) -> 3 terms
    
    # Total new terms: 4 (cross) + 3 + 3 + 3 + 3 + 3 = 19.
    # Total rank = 7 + 19 = 26.
    
    # Let's assign columns 7-25.
    
    # Col 7: A02 * B02 (a0 b0) -> C00
    U[2][7] = 1 # A02 is row 0, col 2? No.
    # A indices: i*m+j.
    # A02 is i=0, j=2 -> U[0*3+2] = U[2].
    # B indices: j*p+k.
    # B02 is j=0, k=2 -> V[0*3+2] = V[2].
    # C00 is k=0, i=0 -> W[0].
    U[2][7] = 1
    V[2][7] = 1
    W[0][7] = 1
    
    # Col 8: A02 * B12 (a0 b1) -> C01
    # A02 is U[2].
    # B12 is j=1, k=2 -> V[1*3+2] = V[5].
    # C01 is k=0, i=1 -> W[1].
    U[2][8] = 1
    V[5][8] = 1
    W[1][8] = 1
    
    # Col 9: A12 * B02 (a1 b0) -> C10
    # A12 is i=1, j=2 -> U[1*3+2] = U[5].
    # B02 is V[2].
    # C10 is k=1, i=0 -> W[3].
    U[5][9] = 1
    V[2][9] = 1
    W[3][9] = 1
    
    # Col 10: A12 * B12 (a1 b1) -> C11
    # A12 is U[5].
    # B12 is V[5].
    # C11 is k=1, i=1 -> W[4].
    U[5][10] = 1
    V[5][10] = 1
    W[4][10] = 1
    
    # Col 11: A20 * B00 -> C20
    # A20 is i=2, j=0 -> U[6].
    # B00 is V[0].
    # C20 is k=2, i=0 -> W[6].
    U[6][11] = 1
    V[0][11] = 1
    W[6][11] = 1
    
    # Col 12: A21 * B00 -> C20
    # A21 is i=2, j=1 -> U[7].
    # B00 is V[0].
    # C20 is W[6].
    U[7][12] = 1
    V[0][12] = 1
    W[6][12] = 1
    
    # Col 13: A22 * B00 -> C20
    # A22 is i=2, j=2 -> U[8].
    # B00 is V[0].
    # C20 is W[6].
    U[8][13] = 1
    V[0][13] = 1
    W[6][13] = 1
    
    # Col 14: A20 * B01 -> C21
    # A20 is U[6].
    # B01 is V[1].
    # C21 is k=2, i=1 -> W[7].
    U[6][14] = 1
    V[1][14] = 1
    W[7][14] = 1
    
    # Col 15: A21 * B01 -> C21
    # A21 is U[7].
    # B01 is V[1].
    # C21 is W[7].
    U[7][15] = 1
    V[1][15] = 1
    W[7][15] = 1
    
    # Col 16: A22 * B01 -> C21
    # A22 is U[8].
    # B01 is V[1].
    # C21 is W[7].
    U[8][16] = 1
    V[1][16] = 1
    W[7][16] = 1
    
    # Col 17: A20 * B02 -> C22
    # A20 is U[6].
    # B02 is V[2].
    # C22 is k=2, i=2 -> W[8].
    U[6][17] = 1
    V[2][17] = 1
    W[8][17] = 1
    
    # Col 18: A21 * B02 -> C22
    # A21 is U[7].
    # B02 is V[2].
    # C22 is W[8].
    U[7][18] = 1
    V[2][18] = 1
    W[8][18] = 1
    
    # Col 19: A22 * B02 -> C22
    # A22 is U[8].
    # B02 is V[2].
    # C22 is W[8].
    U[8][19] = 1
    V[2][19] = 1
    W[8][19] = 1
    
    # Col 20: A00 * B20 -> C02
    # A00 is U[0].
    # B20 is j=2, k=0 -> V[2*3+0] = V[6].
    # C02 is k=0, i=2 -> W[2].
    U[0][20] = 1
    V[6][20] = 1
    W[2][20] = 1
    
    # Col 21: A01 * B20 -> C02
    # A01 is U[1].
    # B20 is V[6].
    # C02 is W[2].
    U[1][21] = 1
    V[6][21] = 1
    W[2][21] = 1
    
    # Col 22: A02 * B20 -> C02
    # A02 is U[2].
    # B20 is V[6].
    # C02 is W[2].
    U[2][22] = 1
    V[6][22] = 1
    W[2][22] = 1
    
    # Col 23: A10 * B20 -> C12
    # A10 is U[3].
    # B20 is V[6].
    # C12 is k=1, i=2 -> W[5].
    U[3][23] = 1
    V[6][23] = 1
    W[5][23] = 1
    
    # Col 24: A11 * B20 -> C12
    # A11 is U[4].
    # B20 is V[6].
    # C12 is W[5].
    U[4][24] = 1
    V[6][24] = 1
    W[5][24] = 1
    
    # Col 25: A12 * B20 -> C12
    # A12 is U[5].
    # B20 is V[6].
    # C12 is W[5].
    U[5][25] = 1
    V[6][25] = 1
    W[5][25] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
