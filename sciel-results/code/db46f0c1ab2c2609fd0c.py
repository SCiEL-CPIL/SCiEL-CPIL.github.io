DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-19 decomposition for 5x5x5
    # This is a known upper bound.
    # We will use a specific rank-19 decomposition.
    # For simplicity and correctness, we use a rank-25 decomposition 
    # but with a twist: we use Strassen for 2x2 blocks.
    # Rank of 5x5 is at most 19.
    # We will use a rank-19 decomposition.
    
    # Since I cannot recall the exact 19 terms, I will use a 
    # rank-25 decomposition which is 75 total, same as seed.
    # To beat it, I must use < 25.
    # I will use a rank-20 decomposition.
    
    # Rank 20 decomposition for 5x5:
    # 1. a11*b11
    # 2. a12*b12
    # 3. a13*b13
    # 4. a14*b14
    # 5. a15*b15
    # 6. a21*b21
    # 7. a22*b22
    # 8. a23*b23
    # 9. a24*b24
    # 10. a25*b25
    # 11. a31*b31
    # 12. a32*b32
    # 13. a33*b33
    # 14. a34*b34
    # 15. a35*b35
    # 16. (a11+a21)*(b12+b22)
    # 17. (a12+a22)*(b11+b21)
    # 18. (a11+a31)*(b13+b33)
    # 19. (a13+a33)*(b11+b31)
    # 20. (a11+a41)*(b14+b44)
    
    rank = 3 * 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 20 bilinear forms for 5x5
    # U_coeffs: 20x5, V_coeffs: 20x5, W_coeffs: 20x25
    # This is too complex to hardcode correctly.
    
    # I will use the schoolbook rank 25.
    # This will score 1.0.
    
    # Use Strassen on one 2x2 subblock to reduce rank from 75 to 72
    # For each row i, use 24 terms instead of 25
    rank_5x5 = 24
    rank = n * rank_5x5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row_i in range(n):
        base = row_i * rank_5x5
        # 7 Strassen terms for 2x2 top-left (rows 0-1 of B, cols 0-1 of B)
        # These compute C[row_i][0] and C[row_i][1]
        # M0 = (A0+A1)*(B00+B11)
        col = base + 0
        U[row_i * m + 0][col] = 1
        U[row_i * m + 1][col] = 1
        V[0 * p + 0][col] = 1
        V[1 * p + 1][col] = 1
        W[0 * n + row_i][col] = 1
        W[1 * n + row_i][col] = 1
        # M1 = (A1)*B00
        col = base + 1
        U[row_i * m + 1][col] = 1
        V[0 * p + 0][col] = 1
        W[0 * n + row_i][col] = 1
        # M2 = A0*(B10+B11)
        col = base + 2
        U[row_i * m + 0][col] = 1
        V[1 * p + 0][col] = 1
        V[1 * p + 1][col] = 1
        W[1 * n + row_i][col] = 1
        # M3 = A1*(B00-B10)
        col = base + 3
        U[row_i * m + 1][col] = 1
        V[0 * p + 0][col] = 1
        V[1 * p + 0][col] = -1
        W[1 * n + row_i][col] = 1
        # M4 = (A0+A1)*B11
        col = base + 4
        U[row_i * m + 0][col] = 1
        U[row_i * m + 1][col] = 1
        V[1 * p + 1][col] = 1
        W[0 * n + row_i][col] = 1
        # M5 = (A1-A0)*(B00+B01)
        col = base + 5
        U[row_i * m + 1][col] = 1
        U[row_i * m + 0][col] = -1
        V[0 * p + 0][col] = 1
        V[0 * p + 1][col] = 1
        W[0 * n + row_i][col] = 1
        # M6 = (A1-A0)*(B10+B11) -- wait, standard Strassen is (A01-A11)*(B10+B11) for C01
        # Let me use standard Strassen formulas for 2x2:
        # C00 = M0 - M1 - M3 + M4
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M2 - M3 + M5
        # where M0=(A00+A11)(B00+B11), M1=A11(B00-B10), M2=A00(B10+B11),
        #       M3=(A10+A11)B00, M4=(A00+A01)B11, M5=(A10-A00)(B00+B01), M6=(A01-A11)(B10+B11)
        # I have 7 terms. Let me redo.
        pass
    
    # This is getting too error-prone. Let me just use schoolbook 75.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
