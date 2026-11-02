"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Total dimensions
    dim_A = n * m  # 12
    dim_B = m * p  # 32
    dim_C = p * n  # 24
    
    # We will use a constructive algorithm based on Strassen for 2x2 blocks.
    # Rank estimate:
    # Split A (3x4) into A1 (2x4) and A2 (1x4).
    # Split B (4x8) into B1 (4x4) and B2 (4x4).
    # C = [A1*B1 + A2*B1, A1*B2 + A2*B2]
    
    # A2 (1x4) * B (4x8) -> 8 dot products of length 4. Rank 8.
    # A1 (2x4) * B1 (4x4) -> 4 multiplications of 2x2 matrices.
    # Each 2x2 mult has rank 7 (Strassen).
    # So A1*B1 has rank 4 * 7 = 28.
    # A1*B2 has rank 4 * 7 = 28.
    # Total rank = 8 + 28 + 28 = 64.
    
    R = 64
    U = [[0.0] * R for _ in range(dim_A)]
    V = [[0.0] * R for _ in range(dim_B)]
    W = [[0.0] * R for _ in range(dim_C)]
    
    current_rank = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal current_rank
        if current_rank >= R:
            return
        U[u_idx][current_rank] += u_val
        V[v_idx][current_rank] += v_val
        W[w_idx][current_rank] += w_val
        current_rank += 1

    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 multiplication.
        A is at (a_row_offset, a_col_offset) in global A.
        B is at (b_row_offset, b_col_offset) in global B.
        C is at (c_row_offset, c_col_offset) in global C.
        Note: A is 2x2, B is 2x2, C is 2x2.
        Global A index: (a_row_offset + i) * m + (a_col_offset + j)
        Global B index: (b_row_offset + i) * p + (b_col_offset + j)
        Global C index: (c_col_offset + j) * n + (c_row_offset + i)  # W[k*n+i]
        """
        # Strassen terms for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the local indices (i,j) to global indices.
        
        def get_a_idx(i, j):
            return (a_row_offset + i) * m + (a_col_offset + j)
            
        def get_b_idx(i, j):
            return (b_row_offset + i) * p + (b_col_offset + j)
            
        def get_c_idx(i, j):
            # C is 3x8. W index is k*n + i.
            # c_row_offset + i is the row i in C.
            # c_col_offset + j is the col j in C.
            return (c_col_offset + j) * n + (c_row_offset + i)

        # M1
        r = current_rank
        # A parts: A00, A11
        U[get_a_idx(0,0)][r] = 1
        U[get_a_idx(1,1)][r] = 1
        # B parts: B00, B11
        V[get_b_idx(0,0)][r] = 1
        V[get_b_idx(1,1)][r] = 1
        # C parts: C00 (+1), C11 (+1)
        W[get_c_idx(0,0)][r] = 1
        W[get_c_idx(1,1)][r] = 1
        current_rank += 1
        
        # M2
        r = current_rank
        # A parts: A10, A11
        U[get_a_idx(1,0)][r] = 1
        U[get_a_idx(1,1)][r] = 1
        # B parts: B00
        V[get_b_idx(0,0)][r] = 1
        # C parts: C10 (+1), C11 (-1)
        W[get_c_idx(1,0)][r] = 1
        W[get_c_idx(1,1)][r] = -1
        current_rank += 1
        
        # M3
        r = current_rank
        # A parts: A00
        U[get_a_idx(0,0)][r] = 1
        # B parts: B01, B11
        V[get_b_idx(0,1)][r] = 1
        V[get_b_idx(1,1)][r] = -1
        # C parts: C01 (+1), C11 (+1)
        W[get_c_idx(0,1)][r] = 1
        W[get_c_idx(1,1)][r] = 1
        current_rank += 1
        
        # M4
        r = current_rank
        # A parts: A11
        U[get_a_idx(1,1)][r] = 1
        # B parts: B10, B00
        V[get_b_idx(1,0)][r] = 1
        V[get_b_idx(0,0)][r] = -1
        # C parts: C00 (+1), C10 (+1)
        W[get_c_idx(0,0)][r] = 1
        W[get_c_idx(1,0)][r] = 1
        current_rank += 1
        
        # M5
        r = current_rank
        # A parts: A00, A01
        U[get_a_idx(0,0)][r] = 1
        U[get_a_idx(0,1)][r] = 1
        # B parts: B11
        V[get_b_idx(1,1)][r] = 1
        # C parts: C00 (-1), C01 (+1)
        W[get_c_idx(0,0)][r] = -1
        W[get_c_idx(0,1)][r] = 1
        current_rank += 1
        
        # M6
        r = current_rank
        # A parts: A10, A00
        U[get_a_idx(1,0)][r] = 1
        U[get_a_idx(0,0)][r] = -1
        # B parts: B00, B01
        V[get_b_idx(0,0)][r] = 1
        V[get_b_idx(0,1)][r] = 1
        # C parts: C11 (+1)
        W[get_c_idx(1,1)][r] = 1
        current_rank += 1
        
        # M7
        r = current_rank
        # A parts: A01, A11
        U[get_a_idx(0,1)][r] = 1
        U[get_a_idx(1,1)][r] = -1
        # B parts: B10, B11
        V[get_b_idx(1,0)][r] = 1
        V[get_b_idx(1,1)][r] = 1
        # C parts: C00 (+1)
        W[get_c_idx(0,0)][r] = 1
        current_rank += 1

    # 1. Handle A1 (rows 0-1) * B1 (cols 0-3)
    # A1 is 2x4. B1 is 4x4.
    # Split A1 into A11 (2x2, cols 0-1) and A12 (2x2, cols 2-3).
    # Split B1 into B11 (2x2, rows 0-1, cols 0-1), B12 (2x2, rows 0-1, cols 2-3),
    #              B21 (2x2, rows 2-3, cols 0-1), B22 (2x2, rows 2-3, cols 2-3).
    
    # C_left_top = A11*B11 + A12*B21
    # C_right_top = A11*B12 + A12*B22
    
    # A11 * B11
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    # A12 * B21
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # A11 * B12
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    # A12 * B22
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # 2. Handle A1 (rows 0-1) * B2 (cols 4-7)
    # B2 is 4x4.
    # Split B2 into B21 (rows 0-1, cols 4-5), B22 (rows 0-1, cols 6-7),
    #              B23 (rows 2-3, cols 4-5), B24 (rows 2-3, cols 6-7).
    # Note: Naming collision with previous B21/B22. Let's call them B2_left, B2_right etc.
    # B2_left_top (rows 0-1, cols 4-5)
    # B2_right_top (rows 0-1, cols 6-7)
    # B2_left_bot (rows 2-3, cols 4-5)
    # B2_right_bot (rows 2-3, cols 6-7)
    
    # C_left_mid = A11*B2_left_top + A12*B2_left_bot
    # C_right_mid = A11*B2_right_top + A12*B2_right_bot
    
    # A11 * B2_left_top
    add_strassen_2x2(0, 0, 0, 4, 0, 4)
    # A12 * B2_left_bot
    add_strassen_2x2(0, 2, 2, 4, 0, 4)
    
    # A11 * B2_right_top
    add_strassen_2x2(0, 0, 0, 6, 0, 6)
    # A12 * B2_right_bot
    add_strassen_2x2(0, 2, 2, 6, 0, 6)
    
    # 3. Handle A2 (row 2) * B (cols 0-7)
    # A2 is 1x4.
    # B1 is 4x4 (cols 0-3).
    # B2 is 4x4 (cols 4-7).
    
    # A2 * B1
    # A2[j] * B1[j, k]
    # This is 4 dot products of length 4? No.
    # A2 is a row vector.
    # C[2, k] = sum_j A2[j] B[j, k].
    # For each k in 0..3, we have a term.
    # We can treat this as 4 separate bilinear forms?
    # No, it's simpler to just add the terms directly.
    # For each j in 0..3, we have a product A2[j] * B[j, :].
    # This product contributes to C[2, 0], C[2, 1], C[2, 2], C[2, 3].
    # We can create one rank term for each j?
    # If we use one rank term for j=0:
    # U[A2[0]] = 1
    # V[B[0,0]] = 1, V[B[0,1]] = 1, V[B[0,2]] = 1, V[B[0,3]] = 1
    # W[C[2,0]] = 1, W[C[2,1]] = 1, W[C[2,2]] = 1, W[C[2,3]] = 1
    # This works! It is a single rank-1 term that computes the entire contribution of the j-th row of B to the bottom row of C.
    
    # So for A2 * B1 (cols 0-3):
    for j in range(4):
        r = current_rank
        # A part: A[2, j]
        U[2 * m + j][r] = 1
        # B parts: B[j, 0], B[j, 1], B[j, 2], B[j, 3]
        for k in range(4):
            V[j * p + k][r] = 1
        # C parts: C[2, 0], C[2, 1], C[2, 2], C[2, 3]
        for k in range(4):
            W[k * n + 2][r] = 1
        current_rank += 1
        
    # For A2 * B2 (cols 4-7):
    for j in range(4):
        r = current_rank
        # A part: A[2, j]
        U[2 * m + j][r] = 1
        # B parts: B[j, 4], B[j, 5], B[j, 6], B[j, 7]
        for k in range(4, 8):
            V[j * p + k][r] = 1
        # C parts: C[2, 4], C[2, 5], C[2, 6], C[2, 7]
        for k in range(4, 8):
            W[k * n + 2][r] = 1
        current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
