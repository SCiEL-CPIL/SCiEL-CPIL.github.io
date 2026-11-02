import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # The total rank is the sum of the ranks of the individual block multiplications.
    # We partition the matrices as described.
    
    # Strassen factors for 2x2x2
    # C = A B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-d)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U (4x7), V (4x7), W (4x7)
    # Indices: A: 0,1,2,3 (a,b,c,d), B: 0,1,2,3 (e,f,g,h), C: 0,1,2,3 (C11,C12,C21,C22)
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # m2 = (c+d)e
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    
    # m3 = a(f-h)
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[2, 2] = 1
    
    # m4 = d(g-f)
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[1, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[2, 3] = 1
    
    # m5 = (a+b)h
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1
    
    # m6 = (c-d)(e+f)
    U_strassen[2, 5] = 1; U_strassen[3, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    # Schoolbook for 2x1x2
    # A is 2x1 (a, c), B is 1x2 (e, f)
    # C11 = ae, C12 = af, C21 = ce, C22 = cf
    # Rank 4.
    U_sb = np.zeros((2, 4))
    V_sb = np.zeros((1, 4))
    W_sb = np.zeros((2, 4))
    
    # ae
    U_sb[0, 0] = 1
    V_sb[0, 0] = 1
    W_sb[0, 0] = 1
    
    # af
    U_sb[0, 1] = 1
    V_sb[0, 1] = 1
    W_sb[0, 1] = 1
    # Wait, W index for C12 is 1.
    # W_sb[0, 1] corresponds to C11? No.
    # W is (p*n, rank) = (2*2, 4) = (4, 4).
    # W indices: 0: C11, 1: C12, 2: C21, 3: C22.
    
    W_sb = np.zeros((4, 4))
    
    # ae -> C11
    U_sb[0, 0] = 1
    V_sb[0, 0] = 1
    W_sb[0, 0] = 1
    
    # af -> C12
    U_sb[0, 1] = 1
    V_sb[0, 1] = 1
    W_sb[1, 1] = 1
    
    # ce -> C21
    U_sb[1, 2] = 1
    V_sb[0, 2] = 1
    W_sb[2, 2] = 1
    
    # cf -> C22
    U_sb[1, 3] = 1
    V_sb[0, 3] = 1
    W_sb[3, 3] = 1

    # Now assemble the full matrices.
    # Total rank = 12 * 7 + 6 * 4 = 84 + 24 = 108.
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: A11 B11 (2x2 x 2x2)
    # A11 is rows 0-1, cols 0-1 of A. Indices 0-3 in U.
    # B11 is rows 0-1, cols 0-1 of B. Indices 0-3 in V.
    # C11 is rows 0-1, cols 0-1 of C. Indices 0-3 in W.
    U[0:4, col:col+7] = U_strassen
    V[0:4, col:col+7] = V_strassen
    W[0:4, col:col+7] = W_strassen
    col += 7
    
    # Block 2: A12 B21 (2x2 x 2x2)
    # A12 is rows 0-1, cols 2-3 of A. Indices 4-7 in U.
    # B21 is rows 2-3, cols 0-1 of B. Indices 4-7 in V.
    # C11 is rows 0-1, cols 0-1 of C. Indices 0-3 in W.
    U[4:8, col:col+7] = U_strassen
    V[4:8, col:col+7] = V_strassen
    W[0:4, col:col+7] = W_strassen
    col += 7
    
    # Block 3: A13 B31 (2x1 x 1x2)
    # A13 is rows 0-1, col 4 of A. Indices 8-9 in U.
    # B31 is row 4, cols 0-1 of B. Indices 8-9 in V.
    # C11 is rows 0-1, cols 0-1 of C. Indices 0-3 in W.
    U[8:10, col:col+4] = U_sb
    V[8:10, col:col+4] = V_sb
    W[0:4, col:col+4] = W_sb
    col += 4
    
    # Block 4: A11 B12 (2x2 x 2x2)
    # A11: 0-3. B12: rows 0-1, cols 2-3. Indices 2-5 in V.
    # C12: rows 0-1, cols 2-3. Indices 4-7 in W.
    U[0:4, col:col+7] = U_strassen
    V[2:6, col:col+7] = V_strassen
    W[4:8, col:col+7] = W_strassen
    col += 7
    
    # Block 5: A12 B22 (2x2 x 2x2)
    # A12: 4-7. B22: rows 2-3, cols 2-3. Indices 6-9 in V.
    # C12: 4-7.
    U[4:8, col:col+7] = U_strassen
    V[6:10, col:col+7] = V_strassen
    W[4:8, col:col+7] = W_strassen
    col += 7
    
    # Block 6: A13 B32 (2x1 x 1x2)
    # A13: 8-9. B32: row 4, cols 2-3. Indices 10-11 in V.
    # C12: 4-7.
    U[8:10, col:col+4] = U_sb
    V[10:12, col:col+4] = V_sb
    W[4:8, col:col+4] = W_sb
    col += 4
    
    # Block 7: A11 B13 (2x2 x 2x2)
    # A11: 0-3. B13: rows 0-1, cols 4-5. Indices 4-7 in V? No.
    # B cols: 0-1, 2-3, 4-5.
    # B13 is rows 0-1, cols 4-5. Indices 8-11 in V.
    # C13: rows 0-1, cols 4-5. Indices 8-11 in W.
    U[0:4, col:col+7] = U_strassen
    V[8:12, col:col+7] = V_strassen
    W[8:12, col:col+7] = W_strassen
    col += 7
    
    # Block 8: A12 B23 (2x2 x 2x2)
    # A12: 4-7. B23: rows 2-3, cols 4-5. Indices 12-15 in V.
    # C13: 8-11.
    U[4:8, col:col+7] = U_strassen
    V[12:16, col:col+7] = V_strassen
    W[8:12, col:col+7] = W_strassen
    col += 7
    
    # Block 9: A13 B33 (2x1 x 1x2)
    # A13: 8-9. B33: row 4, cols 4-5. Indices 14-15 in V.
    # C13: 8-11.
    U[8:10, col:col+4] = U_sb
    V[14:16, col:col+4] = V_sb
    W[8:12, col:col+4] = W_sb
    col += 4
    
    # Block 10: A21 B11 (2x2 x 2x2)
    # A21: rows 2-3, cols 0-1. Indices 10-13 in U.
    # B11: 0-3.
    # C21: rows 2-3, cols 0-1. Indices 4-7 in W.
    U[10:14, col:col+7] = U_strassen
    V[0:4, col:col+7] = V_strassen
    W[4:8, col:col+7] = W_strassen
    col += 7
    
    # Block 11: A22 B21 (2x2 x 2x2)
    # A22: rows 2-3, cols 2-3. Indices 14-17 in U.
    # B21: 4-7.
    # C21: 4-7.
    U[14:18, col:col+7] = U_strassen
    V[4:8, col:col+7] = V_strassen
    W[4:8, col:col+7] = W_strassen
    col += 7
    
    # Block 12: A23 B31 (2x1 x 1x2)
    # A23: rows 2-3, col 4. Indices 18-19 in U.
    # B31: 8-9.
    # C21: 4-7.
    U[18:20, col:col+4] = U_sb
    V[8:10, col:col+4] = V_sb
    W[4:8, col:col+4] = W_sb
    col += 4
    
    # Block 13: A21 B12 (2x2 x 2x2)
    # A21: 10-13. B12: 2-5.
    # C22: rows 2-3, cols 2-3. Indices 8-11 in W.
    U[10:14, col:col+7] = U_strassen
    V[2:6, col:col+7] = V_strassen
    W[8:12, col:col+7] = W_strassen
    col += 7
    
    # Block 14: A22 B22 (2x2 x 2x2)
    # A22: 14-17. B22: 6-9.
    # C22: 8-11.
    U[14:18, col:col+7] = U_strassen
    V[6:10, col:col+7] = V_strassen
    W[8:12, col:col+7] = W_strassen
    col += 7
    
    # Block 15: A23 B32 (2x1 x 1x2)
    # A23: 18-19. B32: 10-11.
    # C22: 8-11.
    U[18:20, col:col+4] = U_sb
    V[10:12, col:col+4] = V_sb
    W[8:12, col:col+4] = W_sb
    col += 4
    
    # Block 16: A21 B13 (2x2 x 2x2)
    # A21: 10-13. B13: 8-11.
    # C23: rows 2-3, cols 4-5. Indices 12-15 in W.
    U[10:14, col:col+7] = U_strassen
    V[8:12, col:col+7] = V_strassen
    W[12:16, col:col+7] = W_strassen
    col += 7
    
    # Block 17: A22 B23 (2x2 x 2x2)
    # A22: 14-17. B23: 12-15.
    # C23: 12-15.
    U[14:18, col:col+7] = U_strassen
    V[12:16, col:col+7] = V_strassen
    W[12:16, col:col+7] = W_strassen
    col += 7
    
    # Block 18: A23 B33 (2x1 x 1x2)
    # A23: 18-19. B33: 14-15.
    # C23: 12-15.
    U[18:20, col:col+4] = U_sb
    V[14:16, col:col+4] = V_sb
    W[12:16, col:col+4] = W_sb
    col += 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
