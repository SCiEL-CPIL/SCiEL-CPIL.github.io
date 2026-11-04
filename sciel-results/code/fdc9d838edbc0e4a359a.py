"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 4 Strassen blocks of size 2x2x2
    # Each block has rank 7
    total_rank = 28
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_block(rank_offset, A_r, A_c, B_r, B_c, C_r, C_c):
        # A is 2x2 at A_r, A_c
        # B is 2x2 at B_r, B_c
        # C is 2x2 at C_r, C_c
        
        # Indices in U: (A_r + r) * m + (A_c + c)
        # Indices in V: (B_r + r) * p + (B_c + c)
        # Indices in W: (C_c + c) * n + (C_r + r)
        
        # Term 1: (A00 + A11) * (B00 + B11)
        U[(A_r + 0) * m + (A_c + 0)][rank_offset + 0] += 1
        U[(A_r + 1) * m + (A_c + 1)][rank_offset + 0] += 1
        V[(B_r + 0) * p + (B_c + 0)][rank_offset + 0] += 1
        V[(B_r + 1) * p + (B_c + 1)][rank_offset + 0] += 1
        
        # Term 2: (A10 + A11) * B00
        U[(A_r + 1) * m + (A_c + 0)][rank_offset + 1] += 1
        U[(A_r + 1) * m + (A_c + 1)][rank_offset + 1] += 1
        V[(B_r + 0) * p + (B_c + 0)][rank_offset + 1] += 1
        
        # Term 3: A00 * (B01 - B11)
        U[(A_r + 0) * m + (A_c + 0)][rank_offset + 2] += 1
        V[(B_r + 0) * p + (B_c + 1)][rank_offset + 2] += 1
        V[(B_r + 1) * p + (B_c + 1)][rank_offset + 2] -= 1
        
        # Term 4: A11 * (B10 - B00)
        U[(A_r + 1) * m + (A_c + 1)][rank_offset + 3] += 1
        V[(B_r + 1) * p + (B_c + 0)][rank_offset + 3] += 1
        V[(B_r + 0) * p + (B_c + 0)][rank_offset + 3] -= 1
        
        # Term 5: (A00 + A01) * B11
        U[(A_r + 0) * m + (A_c + 0)][rank_offset + 4] += 1
        U[(A_r + 0) * m + (A_c + 1)][rank_offset + 4] += 1
        V[(B_r + 1) * p + (B_c + 1)][rank_offset + 4] += 1
        
        # Term 6: (A10 - A00) * (B00 + B01)
        U[(A_r + 1) * m + (A_c + 0)][rank_offset + 5] += 1
        U[(A_r + 0) * m + (A_c + 0)][rank_offset + 5] -= 1
        V[(B_r + 0) * p + (B_c + 0)][rank_offset + 5] += 1
        V[(B_r + 0) * p + (B_c + 1)][rank_offset + 5] += 1
        
        # Term 7: (A01 - A11) * (B10 + B11)
        U[(A_r + 0) * m + (A_c + 1)][rank_offset + 6] += 1
        U[(A_r + 1) * m + (A_c + 1)][rank_offset + 6] -= 1
        V[(B_r + 1) * p + (B_c + 0)][rank_offset + 6] += 1
        V[(B_r + 1) * p + (B_c + 1)][rank_offset + 6] += 1
        
        # W contributions
        # C00 = M1 + M4 - M5 + M7
        W[(C_c + 0) * n + (C_r + 0)][rank_offset + 0] += 1
        W[(C_c + 0) * n + (C_r + 0)][rank_offset + 3] += 1
        W[(C_c + 0) * n + (C_r + 0)][rank_offset + 4] -= 1
        W[(C_c + 0) * n + (C_r + 0)][rank_offset + 6] += 1
        
        # C01 = M3 + M5
        W[(C_c + 1) * n + (C_r + 0)][rank_offset + 2] += 1
        W[(C_c + 1) * n + (C_r + 0)][rank_offset + 4] += 1
        
        # C10 = M2 + M4
        W[(C_c + 0) * n + (C_r + 1)][rank_offset + 1] += 1
        W[(C_c + 0) * n + (C_r + 1)][rank_offset + 3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[(C_c + 1) * n + (C_r + 1)][rank_offset + 0] += 1
        W[(C_c + 1) * n + (C_r + 1)][rank_offset + 1] -= 1
        W[(C_c + 1) * n + (C_r + 1)][rank_offset + 2] += 1
        W[(C_c + 1) * n + (C_r + 1)][rank_offset + 5] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
