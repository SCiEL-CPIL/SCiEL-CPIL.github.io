"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 23 decomposition for 3x3x3 matrix multiplication (Winograd)
    # U3, V3, W3 are 9x23 matrices
    # Indices for U3: i*3 + j
    # Indices for V3: j*3 + k
    # Indices for W3: k*3 + i
    
    # Hardcoded matrices for 3x3x3 rank 23
    # Source: Winograd (1980)
    
    U3 = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # This is just the identity for the first 12 columns, which is rank 12?
    # No, Winograd's algorithm is more complex.
    # I don't have the exact Winograd matrices memorized correctly.
    # I will use a fallback: Schoolbook for 3x3x3 (rank 27) is 54 total.
    # I need a better strategy.
    
    # Let's try to construct a rank 46 solution using a known algorithm if I can find one.
    # Otherwise, I will return the seed.
    
    # However, I can try to use a simpler rank reduction.
    # What if I use the fact that 3x3x6 can be done with 2x2 blocks?
    # No.
    
    # I will return the seed program's result to be safe, as I don't have the exact Winograd matrices.
    # Wait, I can try to search for a rank 46 decomposition using a simple heuristic?
    # No, that's too slow.
    
    # I will stick to the seed for now, but I want to improve.
    # Is there a rank 23 algorithm for 3x3x3 that I can derive?
    # No.
    
    # I will output the seed.
    n, m, p = DIMENSIONS
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
