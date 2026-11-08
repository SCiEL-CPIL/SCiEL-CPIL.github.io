"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-97 decomposition for 4x4 * 4x7 matrix multiplication.
    Strategy: Strassen-based 4x4x4 (rank 49) for first 4 columns of B,
              plus schoolbook 4x4x3 (rank 48) for last 3 columns of B.
    Total rank = 97.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    # Strassen 2x2x2 base coefficients
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # u_in[l][block]: coefficient of A block in multiplication l
    # Blocks: 0=A00, 1=A01, 2=A10, 3=A11
    u_in = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: (a+b)
        [-1, 0, 1, 0],  # M6: (c-a)
        [0, 1, 0, -1]   # M7: (b-d)
    ]
    # v_in[l][block]: coefficient of B block in multiplication l
    # Blocks: 0=B00, 1=B01, 2=B10, 3=B11
    v_in = [
        [1, 0, 0, 1],   # M1: (e+h)
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: (f-h)
        [-1, 0, 1, 0],  # M4: (g-c)
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: (e+f)
        [0, 0, 1, 1]    # M7: (g+h)
    ]
    # w_out[c_block][l]: coefficient of multiplication l in output block c_block
    # c_block: 0=C00, 1=C01, 2=C10, 3=C11
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = M1+M4-M5+M7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M3+M5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M2+M4
        [1, -1, 1, 0, 0, 1, 0]    # C11 = M1-M2+M3+M6
    ]

    # For 4x4x4, we apply Strassen at two levels:
    # Outer: 4x4 = 2x2 blocks of 2x2
    # Inner: each 2x2 multiplication uses Strassen

    # Outer Strassen coefficients (same structure as inner)
    u_outer = u_in
    v_outer = v_in
    w_outer = w_out

    rank = 97
    U = [[0] * rank for _ in range(n * m)]   # 16 x 97
    V = [[0] * rank for _ in range(m * p)]   # 28 x 97
    W = [[0] * rank for _ in range(p * n)]   # 28 x 97

    col = 0

    # Part 1: Strassen 4x4x4 for columns k=0..3 of B
    for k_outer in range(7):
        if time.time() > deadline:
            break
        for l_inner in range(7):
            if time.time() > deadline:
                break

            # U vector: 16 entries (4x4 matrix A)
            # U[i*4+j] for i,j in 0..3
            u_vec = [0] * 16
            for i_in in range(2):
                for j_in in range(2):
                    coeff_in = u_in[l_inner][i_in * 2 + j_in]
                    if coeff_in == 0:
                        continue
                    for b_out in range(4):
                        coeff_out = u_outer[k_outer][b_out]
                        if coeff_out == 0:
                            continue
                        # b_out maps to 2x2 block position in 4x4
                        # b_out=0: rows 0-1, cols 0-1
                        # b_out=1: rows 0-1, cols 2-3
                        # b_out=2: rows 2-3, cols 0-1
                        # b_out=3: rows 2-3, cols 2-3
                        row_block = b_out // 2
                        col_block = b_out % 2
                        i_global = 2 * row_block + i_in
                        j_global = 2 * col_block + j_in
                        u_vec[i_global * 4 + j_global] += coeff_in * coeff_out

            # V vector: 28 entries (4x7 matrix B)
            # V[j*7+k] for j in 0..3, k in 0..6
            # Only first 4 columns (k=0..3) are used in this part
            v_vec = [0] * 28
            for i_in in range(2):
                for j_in in range(2):
                    coeff_in = v_in[l_inner][i_in * 2 + j_in]
                    if coeff_in == 0:
                        continue
                    for b_out in range(4):
                        coeff_out = v_outer[k_outer][b_out]
                        if coeff_out == 0:
                            continue
                        row_block = b_out // 2
                        col_block = b_out % 2
                        j_global = 2 * row_block + i_in
                        k_global = 2 * col_block + j_in  # 0..3
                        v_vec[j_global * 7 + k_global] += coeff_in * coeff_out

            # W vector: 28 entries (7x4 output, stored as W[k*n+i] = W[k*4+i])
            # Only first 4 rows (k=0..3) are used
            w_vec = [0] * 28
            for c_block_out in range(4):
                coeff_w_out = w_outer[c_block_out][k_outer]
                if coeff_w_out == 0:
                    continue
                for c_block_in in range(4):
                    coeff_w_in = w_out[c_block_in][l_inner]
                    if coeff_w_in == 0:
                        continue
                    total_coeff = coeff_w_out * coeff_w_in
                    if total_coeff == 0:
                        continue

                    c_row_block = c_block_out // 2
                    c_col_block = c_block_out % 2
                    c_inner_row = c_block_in // 2
                    c_inner_col = c_block_in % 2

                    i_global = 2 * c_row_block + c_inner_row
                    k_global = 2 * c_col_block + c_inner_col  # 0..3

                    # W[k*4 + i]
                    w_vec[k_global * 4 + i_global] += total_coeff

            # Store
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]

            col += 1

    # Part 2: Schoolbook for 4x4 * 4x3 (right 3 columns of B)
    # Rank = 4*4*3 = 48
    for i in range(4):
        if time.time() > deadline:
            break
        for j in range(4):
            if time.time() > deadline:
                break
            for k_prime in range(3):
                if time.time() > deadline:
                    break
                k_global = 4 + k_prime  # columns 4, 5, 6 of B
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
