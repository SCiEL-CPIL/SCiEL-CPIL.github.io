def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 2: Schoolbook for 4x4 * 4x3
    col = 49
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (4 + k)][col] = 1
                W[(4 + k) * n + i][col] = 1
                col += 1
                
    # Part 1: Strassen for 4x4 * 4x4
    # We generate the 49 terms.
    # I will use a helper function to get the Strassen terms for 2x2x2.
    # Then map them to the 4x4x4 indices.
    
    # Strassen 2x2x2 terms:
    # (i1, j1, k1, coeff)
    # i1, j1, k1 in {0, 1}
    
    # M1: (A11+A22)(B11+B22) -> 4 terms
    # M2: (A21+A22)B11 -> 2 terms
    # M3: A11(B12-B22) -> 2 terms
    # M4: A22(B21-B11) -> 2 terms
    # M5: (A11+A12)B22 -> 2 terms
    # M6: (A21-A11)(B11+B12) -> 4 terms
    # M7: (A12-A22)(B21+B22) -> 4 terms
    # Total 20 terms for 2x2x2? No, 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is 4 scalar multiplications in schoolbook.
    # But Strassen replaces that with 7 scalar multiplications.
    # So for 4x4x4, we have 7 Strassen 2x2x2 decompositions.
    # Each 2x2x2 Strassen decomposition has 7 scalar terms.
    # Total 49 terms.
    
    # I will hardcode the 49 terms.
    
    # This is too long to hardcode manually in the thought process.
    # I will write a function to generate them.
    
    pass
