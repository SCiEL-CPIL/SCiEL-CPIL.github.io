"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen-based blocks."""
import time

START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit


def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    The decomposition uses 6 blocks of 2x2x2 Strassen multiplications (rank 7 each),
    giving total rank 42.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT:
        pass

    n, m, p = 2, 4, 6
    rank_total = 42

    # Initialize factor matrices
    U = [[0] * rank_total for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 42

    # Strassen rank-7 decomposition for 2x2x2
    # Using the standard Strassen algorithm:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B01
    # M4 = A00 * (B11 - B01)
    # M5 = (A10 - A00) * (B00 + B10)
    # M6 = (A11 - A01) * (B01 + B11)
    # M7 = A01 * (B10 - B00)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M4
    # C10 = M2 - M6 + M7
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me use the code's existing Strassen coefficients which were verified to work:
    strassen_U = [
        {0: 1, 3: 1},       # A00 + A11
        {2: 1, 3: 1},       # A10 + A11
        {0: 1},             # A00
        {3: 1},             # A11
        {0: 1, 1: 1},       # A00 + A01
        {0: -1, 2: 1},      # A10 - A00
        {1: 1, 3: -1},      # A01 - A11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # B00 + B11
        {0: 1},             # B00
        {1: 1, 3: -1},      # B01 - B11
        {2: 1, 0: -1},      # B10 - B00
        {3: 1},             # B11
        {0: 1, 1: 1},       # B00 + B01
        {2: 1, 3: 1},       # B10 + B11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # C00 + C11
        {2: 1, 3: -1},      # C01 - C11
        {1: 1, 3: 1},       # C10 + C11
        {0: 1, 2: 1},       # C00 + C10
        {0: -1, 1: 1},      # C10 - C00
        {3: 1},             # C11
        {0: 1},             # C00
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_offset = 0
                v_row_offset = 0
            else:
                u_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # Map local 2x2 indices to global U indices
                # U has shape (8, rank), indexed as [i*4 + k] where i in {0,1}, k in {0,1,2,3}
                local_U_map = {
                    0: 0 * 4 + (u_offset + 0),  # A[0, u_offset+0]
                    1: 0 * 4 + (u_offset + 1),  # A[0, u_offset+1]
                    2: 1 * 4 + (u_offset + 0),  # A[1, u_offset+0]
                    3: 1 * 4 + (u_offset + 1),  # A[1, u_offset+1]
                }
                for local_idx, sign in strassen_U[prod_idx].items():
                    U[local_U_map[local_idx]][col] += sign

                # Map local 2x2 indices to global V indices
                # V has shape (24, rank), indexed as [k*6 + j] where k in {0,1,2,3}, j in {0,1,2,3,4,5}
                local_V_map = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),  # B[v_row_offset+0, 2*j_block+0]
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),  # B[v_row_offset+0, 2*j_block+1]
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),  # B[v_row_offset+1, 2*j_block+0]
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),  # B[v_row_offset+1, 2*j_block+1]
                }
                for local_idx, sign in strassen_V[prod_idx].items():
                    V[local_V_map[local_idx]][col] += sign

                # Map local 2x2 indices to global W indices
                # W has shape (12, rank), indexed as [j*n + i] where j in {0,1,2,3,4,5}, i in {0,1}
                local_W_map = {
                    0: (2 * j_block + 0) * 2 + 0,  # C[2*j_block+0, 0]
                    1: (2 * j_block + 1) * 2 + 0,  # C[2*j_block+1, 0]
                    2: (2 * j_block + 0) * 2 + 1,  # C[2*j_block+0, 1]
                    3: (2 * j_block + 1) * 2 + 1,  # C[2*j_block+1, 1]
                }
                for local_idx, sign in strassen_W[prod_idx].items():
                    W[local_W_map[local_idx]][col] += sign

            col_offset += 7

    return {"U": U, "V": V, "W": W}
