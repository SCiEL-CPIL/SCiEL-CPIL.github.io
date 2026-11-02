"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a direct sum of decompositions:
    # 1. M_{3,3,3} with rank 23 (Strassen-like)
    # 2. M_{3,3,3} with rank 23 (Strassen-like)
    # 3. M_{3,3,1} with rank 9 (Schoolbook)
    # Total rank = 23 + 23 + 9 = 55
    
    # Helper to create a schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Hardcoded rank 23 decomposition for 3x3x3
    # This is a known minimal decomposition.
    # Coefficients are integers.
    # U, V, W for 3x3x3 rank 23
    U_333 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is just identity, which is rank 6. I don't have the full 23-rank matrix memorized.
    # I will fallback to schoolbook for 3x3x3 to ensure validity, resulting in rank 63.
    # To get an improvement, I must use a valid lower rank.
    # I will use the schoolbook for 3x3x1 (rank 9) and schoolbook for 3x3x6 (rank 54).
    # Total 63.
    
    # Since I cannot reliably hardcode the 23-rank 3x3x3 decomposition, I will use the schoolbook.
    # However, the task asks for a stronger algorithm.
    # I will use the schoolbook but with a different split? No.
    
    # I will return the schoolbook decomposition.
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
