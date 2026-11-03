"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    # We use a 2x2 Strassen-based decomposition.
    # Partition A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Partition B (4x8) into 2x4 blocks: B = [[B00, B01], [B10, B11]]
    # where each Bij is 2x4.
    # C (4x8) = [[C00, C01], [C10, C11]] where each Cij is 2x4.
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 * 2x4 multiplication, we use 14 multiplications (7 per 2x2 subblock of B).
    # Total: 4 (A blocks) * 2 (B row blocks) * 2 (B col subblocks) * 7 = 112
    #
    # But wait, we can be smarter. Let's use the full Strassen on the 2x2 blocks
    # treating the 2x4 B as two 2x2 matrices side by side.
    
    # Actually, let me just implement the schoolbook with rank 128 first to ensure correctness,
    # then optimize.
    
    rank = n * m * p  # 128
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
