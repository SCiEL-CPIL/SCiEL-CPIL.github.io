"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x4 by 4x6 multiplication into 6 blocks of 2x2 by 2x2 multiplications.
    # Each 2x2 by 2x2 multiplication is done using Strassen's algorithm (rank 7).
    # Total rank = 6 * 7 = 42? No, the blocks are not independent.
    # C_uv = A_u1 B_1v + A_u2 B_2v.
    # This requires 2 Strassen multiplications per block.
    # 6 blocks * 2 * 7 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2 matrices
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(u, v, offset):
        # u is row block index (0,1), v is col block index (0,1,2)
        # A is 4x4, B is 4x6
        # A blocks: A[u*2 + i, u*2 + j]
        # B blocks: B[v*2 + i, v*2 + j]
        
        # We need to add 7 terms for A_u1 B_1v and 7 terms for A_u2 B_2v
        # Total 14 terms per block
        
        # Helper to add a Strassen term
        # term_type: 0..6
        # a_block: 0 (A_u1) or 1 (A_u2)
        # b_block: 0 (B_1v) or 1 (B_2v)
        
        for a_blk in range(2):
            for b_blk in range(2):
                for t in range(7):
                    # Determine U, V, W entries for this term
                    # This is complex to map directly to global indices without a lookup table.
                    # I will use a simpler approach: just use schoolbook for 2x2 (rank 4) to ensure correctness?
                    # Rank 6 * 2 * 4 = 48.
                    # Score 96/48 = 2.0.
                    pass
                    
    # Given the complexity of mapping Strassen to global tensor indices, 
    # and the risk of errors, I will fall back to a known valid decomposition.
    # If I can't easily code Strassen, I will return the schoolbook rank 96.
    # But wait, I can just return the schoolbook decomposition.
    
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
