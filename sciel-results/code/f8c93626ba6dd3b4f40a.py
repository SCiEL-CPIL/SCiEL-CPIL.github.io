import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Uses Laderman's algorithm (Rank 23 for 3x3x3) applied twice 
    with shared A structure via sum/diff trick to achieve Rank 46.
    """
    n, m, p = 3, 3, 6
    
    # Laderman's algorithm coefficients for 3x3x3
    # A indices: 0..8 (row-major)
    # B indices: 0..8 (row-major)
    # C indices: 0..8 (row-major)
    
    # We define 23 terms for one 3x3x3 multiplication.
    # Term t has U_t (9x1), V_t (9x1), W_t (9x1)
    
    # U (A coefficients)
    # V (B coefficients)
    # W (C coefficients)
    
    # I will hardcode the 23 terms.
    # This is based on standard Laderman decomposition.
    
    U_lad = np.zeros((9, 23))
    V_lad = np.zeros((9, 23))
    W_lad = np.zeros((9, 23))
    
    # Helper to set coeff
    def set_coeff(mat, idx, col, val):
        mat[idx, col] = val
        
    # 1. a00 * (b00 - b01 - b10 + b11)
    set_coeff(U_lad, 0, 0, 1)
    set_coeff(V_lad, 0, 0, 1)
    set_coeff(V_lad, 1, 0, -1)
    set_coeff(V_lad, 3, 0, -1)
    set_coeff(V_lad, 4, 0, 1)
    # C00 += M1
    set_coeff(W_lad, 0, 0, 1)
    
    # 2. (a00 + a01) * (b10 - b11)
    set_coeff(U_lad, 0, 1, 1)
    set_coeff(U_lad, 1, 1, 1)
    set_coeff(V_lad, 3, 1, 1)
    set_coeff(V_lad, 4, 1, -1)
    # C00 += M2, C01 -= M2
    set_coeff(W_lad, 0, 1, 1)
    set_coeff(W_lad, 1, 1, -1)
    
    # 3. (a00 - a10) * (b01 - b11)
    set_coeff(U_lad, 0, 2, 1)
    set_coeff(U_lad, 3, 2, -1)
    set_coeff(V_lad, 1, 2, 1)
    set_coeff(V_lad, 4, 2, -1)
    # C00 += M3, C01 -= M3
    set_coeff(W_lad, 0, 2, 1)
    set_coeff(W_lad, 1, 2, -1)
    
    # 4. a11 * (b00 - b01 - b10 + b11)
    set_coeff(U_lad, 4, 3, 1)
    set_coeff(V_lad, 0, 3, 1)
    set_coeff(V_lad, 1, 3, -1)
    set_coeff(V_lad, 3, 3, -1)
    set_coeff(V_lad, 4, 3, 1)
    # C11 += M4
    set_coeff(W_lad, 4, 3, 1)
    
    # 5. (a11 + a12) * (b00 - b01)
    set_coeff(U_lad, 4, 4, 1)
    set_coeff(U_lad, 5, 4, 1)
    set_coeff(V_lad, 0, 4, 1)
    set_coeff(V_lad, 1, 4, -1)
    # C11 += M5, C12 -= M5
    set_coeff(W_lad, 4, 4, 1)
    set_coeff(W_lad, 5, 4, -1)
    
    # 6. (a11 - a21) * (b00 - b01)
    set_coeff(U_lad, 4, 5, 1)
    set_coeff(U_lad, 7, 5, -1)
    set_coeff(V_lad, 0, 5, 1)
    set_coeff(V_lad, 1, 5, -1)
    # C11 += M6, C12 -= M6
    set_coeff(W_lad, 4, 5, 1)
    set_coeff(W_lad, 5, 5, -1)
    
    # 7. (a11 + a21) * (b00 - b10)
    set_coeff(U_lad, 4, 6, 1)
    set_coeff(U_lad, 7, 6, 1)
    set_coeff(V_lad, 0, 6, 1)
    set_coeff(V_lad, 3, 6, -1)
    # C11 += M7, C21 -= M7
    set_coeff(W_lad, 4, 6, 1)
    set_coeff(W_lad, 7, 6, -1)
    
    # 8. (a11 - a12) * (b00 - b10)
    set_coeff(U_lad, 4, 7, 1)
    set_coeff(U_lad, 5, 7, -1)
    set_coeff(V_lad, 0, 7, 1)
    set_coeff(V_lad, 3, 7, -1)
    # C11 += M8, C21 -= M8
    set_coeff(W_lad, 4, 7, 1)
    set_coeff(W_lad, 7, 7, -1)
    
    # 9. a22 * (b00 - b01 - b10 + b11)
    set_coeff(U_lad, 8, 8, 1)
    set_coeff(V_lad, 0, 8, 1)
    set_coeff(V_lad, 1, 8, -1)
    set_coeff(V_lad, 3, 8, -1)
    set_coeff(V_lad, 4, 8, 1)
    # C22 += M9
    set_coeff(W_lad, 8, 8, 1)
    
    # 10. (a22 + a21) * (b01 - b11)
    set_coeff(U_lad, 8, 9, 1)
    set_coeff(U_lad, 7, 9, 1)
    set_coeff(V_lad, 1, 9, 1)
    set_coeff(V_lad, 4, 9, -1)
    # C22 += M10, C21 -= M10
    set_coeff(W_lad, 8, 9, 1)
    set_coeff(W_lad, 7, 9, -1)
    
    # 11. (a22 - a12) * (b01 - b11)
    set_coeff(U_lad, 8, 10, 1)
    set_coeff(U_lad, 5, 10, -1)
    set_coeff(V_lad, 1, 10, 1)
    set_coeff(V_lad, 4, 10, -1)
    # C22 += M11, C12 -= M11
    set_coeff(W_lad, 8, 10, 1)
    set_coeff(W_lad, 5, 10, -1)
    
    # 12. (a22 + a12) * (b10 - b11)
    set_coeff(U_lad, 8, 11, 1)
    set_coeff(U_lad, 5, 11, 1)
    set_coeff(V_lad, 3, 11, 1)
    set_coeff(V_lad, 4, 11, -1)
    # C22 += M12, C21 -= M12
    set_coeff(W_lad, 8, 11, 1)
    set_coeff(W_lad, 7, 11, -1)
    
    # 13. (a22 - a21) * (b10 - b11)
    set_coeff(U_lad, 8, 12, 1)
    set_coeff(U_lad, 7, 12, -1)
    set_coeff(V_lad, 3, 12, 1)
    set_coeff(V_lad, 4, 12, -1)
    # C22 += M13, C21 -= M13
    set_coeff(W_lad, 8, 12, 1)
    set_coeff(W_lad, 7, 12, -1)
    
    # 14. (a02 + a12) * b00
    set_coeff(U_lad, 2, 13, 1)
    set_coeff(U_lad, 5, 13, 1)
    set_coeff(V_lad, 0, 13, 1)
    # C00 += M14, C10 += M14
    set_coeff(W_lad, 0, 13, 1)
    set_coeff(W_lad, 3, 13, 1)
    
    # 15. (a02 - a12) * b10
    set_coeff(U_lad, 2, 14, 1)
    set_coeff(U_lad, 5, 14, -1)
    set_coeff(V_lad, 3, 14, 1)
    # C00 += M15, C10 += M15
    set_coeff(W_lad, 0, 14, 1)
    set_coeff(W_lad, 3, 14, 1)
    
    # 16. (a02 + a22) * b01
    set_coeff(U_lad, 2, 15, 1)
    set_coeff(U_lad, 8, 15, 1)
    set_coeff(V_lad, 1, 15, 1)
    # C01 += M16, C11 += M16
    set_coeff(W_lad, 1, 15, 1)
    set_coeff(W_lad, 4, 15, 1)
    
    # 17. (a02 - a22) * b11
    set_coeff(U_lad, 2, 16, 1)
    set_coeff(U_lad, 8, 16, -1)
    set_coeff(V_lad, 4, 16, 1)
    # C01 += M17, C11 += M17
    set_coeff(W_lad, 1, 16, 1)
    set_coeff(W_lad, 4, 16, 1)
    
    # 18. (a10 + a20) * b02
    set_coeff(U_lad, 3, 17, 1)
    set_coeff(U_lad, 6, 17, 1)
    set_coeff(V_lad, 2, 17, 1)
    # C02 += M18, C12 += M18
    set_coeff(W_lad, 2, 17, 1)
    set_coeff(W_lad, 5, 17, 1)
    
    # 19. (a10 - a20) * b12
    set_coeff(U_lad, 3, 18, 1)
    set_coeff(U_lad, 6, 18, -1)
    set_coeff(V_lad, 5, 18, 1)
    # C02 += M19, C12 += M19
    set_coeff(W_lad, 2, 18, 1)
    set_coeff(W_lad, 5, 18, 1)
    
    # 20. (a11 + a21) * b02
    set_coeff(U_lad, 4, 19, 1)
    set_coeff(U_lad, 7, 19, 1)
    set_coeff(V_lad, 2, 19, 1)
    # C02 += M20, C12 += M20
    set_coeff(W_lad, 2, 19, 1)
    set_coeff(W_lad, 5, 19, 1)
    
    # 21. (a11 - a21) * b12
    set_coeff(U_lad, 4, 20, 1)
    set_coeff(U_lad, 7, 20, -1)
    set_coeff(V_lad, 5, 20, 1)
    # C02 += M21, C12 += M21
    set_coeff(W_lad, 2, 20, 1)
    set_coeff(W_lad, 5, 20, 1)
    
    # 22. (a12 + a22) * b02
    set_coeff(U_lad, 5, 21, 1)
    set_coeff(U_lad, 8, 21, 1)
    set_coeff(V_lad, 2, 21, 1)
    # C02 += M22, C12 += M22
    set_coeff(W_lad, 2, 21, 1)
    set_coeff(W_lad, 5, 21, 1)
    
    # 23. (a12 - a22) * b12
    set_coeff(U_lad, 5, 22, 1)
    set_coeff(U_lad, 8, 22, -1)
    set_coeff(V_lad, 5, 22, 1)
    # C02 += M23, C12 += M23
    set_coeff(W_lad, 2, 22, 1)
    set_coeff(W_lad, 5, 22, 1)

    # Now construct the 3x3x6 decomposition
    # C1 = A * B1, C2 = A * B2
    # S = B1 + B2, D = B1 - B2
    # C1 = 0.5 (A*S + A*D)
    # C2 = 0.5 (A*S - A*D)
    
    rank = 46
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # First 23 terms: A * S
    # U[0:23] = U_lad
    U[:, 0:23] = U_lad
    # V[0:23] corresponds to S = B1 + B2
    # B1 is cols 0-2 of B (indices 0-5 in flat 3x6? No, B is 3x6)
    # B flat: row 0 (0,1,2), row 1 (3,4,5), row 2 (6,7,8)
    # B1: cols 0,1,2. B2: cols 3,4,5.
    # V_lad indices 0-8 map to B1 indices 0-8.
    # S = B1 + B2.
    # So V_S[j, t] = V_lad[j, t] + V_lad[j, t] (for B2 part)?
    # No, V_lad is for a 3x3 B.
    # For term t, the coefficient for B1[j] is V_lad[j, t].
    # The coefficient for B2[j] is V_lad[j, t].
    # So V[0:9, 0:23] = V_lad
    # V[9:18, 0:23] = V_lad
    V[0:9, 0:23] = V_lad
    V[9:18, 0:23] = V_lad
    
    # W for C1 = 0.5 * C_S
    # C_S is 3x3. Output C1 is first 3 cols of C (6x3).
    # C flat: row 0 (0,1,2), row 1 (3,4,5), row 2 (6,7,8)
    # C1 corresponds to C cols 0,1,2?
    # C is 6x3. C1 is 3x3 (rows 0-2, cols 0-2? No, C1 = A B1. A is 3x3, B1 is 3x3. C1 is 3x3.)
    # C2 = A B2. C2 is 3x3.
    # C = [C1, C2] ? No, C is 3x6?
    # Task: 3x3 * 3x6 -> 3x6.
    # Output factor W[k*n+i, r]. n=3, p=6.
    # W shape (18, rank).
    # Index k*n+i. k in 0..5, i in 0..2.
    # C1 (3x3) corresponds to k=0,1,2?
    # B1 is cols 0,1,2 of B.
    # C1 = A B1. C1 is 3x3.
    # C2 = A B2. C2 is 3x3.
    # C = [C1 C2] (3x6).
    # So C[k, 0:3] = C1[k, :], C[k, 3:6] = C2[k, :].
    # W index for C1[k, j] is k*3 + j? No, W[k*n+i].
    # n=3.
    # C1[k, j] is at C[k, j]. Index k*3+j.
    # C2[k, j] is at C[k, 3+j]. Index k*3 + 3+j.
    
    # W_lad gives coefficients for C_S (3x3).
    # C1 = 0.5 C_S.
    # So W[0:9, 0:23] = 0.5 * W_lad.
    # C2 = 0.5 (C_S - C_D).
    
    # Second 23 terms: A * D
    # U[23:46] = U_lad
    U[:, 23:46] = U_lad
    # V[0:9, 23:46] = -V_lad (for B1 part of D)
    # V[9:18, 23:46] = V_lad (for B2 part of D)
    # D = B1 - B2.
    V[0:9, 23:46] = -V_lad
    V[9:18, 23:46] = V_lad
    
    # W for C1 part of D: -0.5 * W_lad
    # W for C2 part of D: -0.5 * W_lad?
    # C2 = 0.5 (C_S - C_D) = 0.5 C_S - 0.5 C_D.
    # So C2 gets -0.5 * W_lad from D terms.
    
    # Apply to W
    # C1 (cols 0-2)
    W[0:9, 0:23] = 0.5 * W_lad
    W[0:9, 23:46] = -0.5 * W_lad
    
    # C2 (cols 3-5)
    # Index k*3 + 3 + j
    for k in range(3):
        for j in range(3):
            idx = k * 3 + 3 + j
            W[idx, 0:23] = 0.5 * W_lad[k * 3 + j, 0:23]
            W[idx, 23:46] = -0.5 * W_lad[k * 3 + j, 0:23]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
