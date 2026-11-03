"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 3x3x3 multiplication (k=0,1,2) using BCLR rank 23
    # BCLR decomposition matrices for 3x3x3
    # U1, V1, W1 are 9x23
    U1 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is just identity, not BCLR. I need the actual BCLR matrices.
    # Since I don't have them memorized, I will use a valid decomposition of rank 27?
    # Or I will use the schoolbook for 3x3x3 (rank 27) and schoolbook for 3x3x2 (rank 18).
    # Total 45. No improvement.
    
    # Let's try to find a valid improvement.
    # If I use Strassen for 2x2x2 (rank 7) and pad to 3x3x3?
    # 3x3x3 can be embedded in 4x4x4.
    # Laderman 4x4x4 rank 16.
    # If I use Laderman for the top-left 4x4x4 and ignore the rest?
    # No, dimensions are fixed.
    
    # I will use the schoolbook decomposition for the whole 3x3x5.
    # Rank 45.
    # To improve, I need a known low-rank decomposition.
    # Without the specific matrices, I cannot guarantee correctness.
    # I will return the seed program's logic but optimized for speed?
    # The seed is already O(1).
    
    # I will stick to the seed program to ensure validity, as I cannot recall the BCLR matrices.
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
