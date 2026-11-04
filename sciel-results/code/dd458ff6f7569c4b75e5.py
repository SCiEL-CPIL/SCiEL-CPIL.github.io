"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    # Use Strassen for the 2x2x2 blocks and schoolbook for the rest.
    # Decompose 2x3x9 as 3 groups of 2x3x3.
    # Each 2x3x3 can be split: rows of A (2), cols of B (3).
    # We'll use a direct construction.
    #
    # Actually, let me use a simpler approach: Strassen for 2x2 blocks.
    # 2x3x9: think of it as 2x3x(3*3). 
    # Better: split the 9 columns into three groups of 3.
    # For each 2x3x3, we can further split the 3 columns of B into (2,1).
    # 2x3x2: split 3 rows of B? No, B is 3x2 here.
    # Let me just use schoolbook for simplicity and correctness, but try to be smart.
    #
    # Actually, the task asks for rank-minimal. Let me use a known result.
    # R(2,3,9) <= ?
    # Let's try: R(2,3,9) = R(2,3,3) * 3 if we can decompose independently.
    # R(2,3,3): schoolbook is 18. Can we do better?
    # R(2,2,2) = 7 (Strassen).
    # R(2,3,3) = R(2,3,2) + R(2,3,1) = ?
    # R(2,3,2): split B cols into 2+0? No.
    # R(2,3,2) = sum over k=0,1 of R(2,3,1)? No, that's not how it works.
    #
    # Let me just implement schoolbook correctly first, then optimize.
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
