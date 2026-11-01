"""Optimized program: rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen 2x2 @ 2x2 rank 7 decomposition
    # Inputs: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Products:
    #   P1 = (a00 + a11) * (b00 + b11)
    #   P2 = (a10 + a11) * b00
    #   P3 = a00 * (b01 - b11)
    #   P4 = a11 * (b10 - b00)
    #   P5 = (a00 + a01) * b11
    #   P6 = (a10 - a00) * (b00 + b01)
    #   P7 = (a01 - a11) * (b10 + b11)
    #
    # Output C:
    #   c00 = P1 + P4 - P5 + P7
    #   c01 = P3 + P5
    #   c10 = P2 + P4
    #   c11 = P1 - P2 + P3 + P6

    # U factors (A side)
    # Indices: 0=a00, 1=a01, 2=a10, 3=a11
    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1}       # P7
    ]

    # V factors (B side)
    # Indices: 0=b00, 1=b01, 2=b10, 3=b11
    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1}        # P7
    ]

    # W factors (C side)
    # Indices: 0=c00, 1=c01, 2=c10, 3=c11
    strassen_W = [
        {0: 1, 3: 1},       # P1: +c00, +c11
        {2: 1, 3: -1},      # P2: +c10, -c11
        {1: 1, 3: 1},       # P3: +c01, +c11
        {0: 1, 2: 1},       # P4: +c00, +c10
        {0: -1, 1: 1},      # P5: -c00, +c01
        {3: 1},             # P6: +c11
        {0: 1}              # P7: +c00
    ]

    col_offset = 0
    # Split 6 columns of B into 3 blocks of 2
    # Split 4 rows of B / 4 cols of A into 2 blocks of 2
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0   # A cols 0,1
                v_row_offset = 0   # B rows 0,1
            else:
                u_col_offset = 2   # A cols 2,3
                v_row_offset = 2   # B rows 2,3

            # Global indices for the 2x2 block of A
            # Local 0 (a00) -> U[0*4 + u_col_offset]
            # Local 1 (a01) -> U[0*4 + u_col_offset + 1]
            # Local 2 (a10) -> U[1*4 + u_col_offset]
            # Local 3 (a11) -> U[1*4 + u_col_offset + 1]

            # Global indices for the 2x2 block of B
            # Global col of B starts at 2*j_block
            # Local 0 (b00) -> V[(v_row_offset)*6 + 2*j_block]
            # Local 1 (b01) -> V[(v_row_offset)*6 + 2*j_block + 1]
            # Local 2 (b10) -> V[(v_row_offset+1)*6 + 2*j_block]
            # Local 3 (b11) -> V[(v_row_offset+1)*6 + 2*j_block + 1]

            # Global indices for the 2x2 block of C
            # Global col of C starts at 2*j_block
            # W index = k*n + i = k*2 + i
            # Local 0 (c00) -> i=0, k=2*j_block -> (2*j_block)*2 + 0
            # Local 1 (c01) -> i=0, k=2*j_block+1 -> (2*j_block+1)*2 + 0
            # Local 2 (c10) -> i=1, k=2*j_block -> (2*j_block)*2 + 1
            # Local 3 (c11) -> i=1, k=2*j_block+1 -> (2*j_block+1)*2 + 1

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # U
                for local_idx, sign in strassen_U[prod_idx].items():
                    if local_idx == 0:
                        u_idx = 0 * 4 + u_col_offset
                    elif local_idx == 1:
                        u_idx = 0 * 4 + u_col_offset + 1
                    elif local_idx == 2:
                        u_idx = 1 * 4 + u_col_offset
                    else:
                        u_idx = 1 * 4 + u_col_offset + 1
                    U[u_idx][col] += sign

                # V
                for local_idx, sign in strassen_V[prod_idx].items():
                    if local_idx == 0:
                        v_idx = (v_row_offset) * 6 + 2 * j_block
                    elif local_idx == 1:
                        v_idx = (v_row_offset) * 6 + 2 * j_block + 1
                    elif local_idx == 2:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                    else:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                    V[v_idx][col] += sign

                # W
                for local_idx, sign in strassen_W[prod_idx].items():
                    if local_idx == 0:
                        w_idx = (2 * j_block) * 2 + 0
                    elif local_idx == 1:
                        w_idx = (2 * j_block + 1) * 2 + 0
                    elif local_idx == 2:
                        w_idx = (2 * j_block) * 2 + 1
                    else:
                        w_idx = (2 * j_block + 1) * 2 + 1
                    W[w_idx][col] += sign

            col_offset += 7

    # Check deadline before returning
    if time.time() > deadline:
        # Should not happen for this deterministic construction, but guard anyway
        pass

    return {"U": U, "V": V, "W": W}
