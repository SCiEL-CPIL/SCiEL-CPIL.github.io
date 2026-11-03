"""Matrix multiplication 2x4 by 4x6 using Strassen-based rank-42 decomposition."""
import time

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 6

    # Rank-42 decomposition: 6 blocks of 2x2 Strassen (rank 7 each)
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 42)

    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # For A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Strassen coefficients: (u_coeffs, v_coeffs, w_coeffs)
    # Local indices: A: 0=A11, 1=A12, 2=A21, 3=A22
    #                B: 0=B11, 1=B12, 2=B21, 3=B22
    #                C: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        # M1: (A11+A22)*(B11+B22) -> C11, C22
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (A21+A22)*B11 -> C21, C22
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: A11*(B12-B22) -> C12, C22
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4: A22*(B21-B11) -> C11, C21
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (A11+A12)*B22 -> C11, C12
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6: (A21-A11)*(B11+B12) -> C22
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7: (A12-A22)*(B21+B22) -> C11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    # Define 6 blocks: each is a 2x2 A-block times a 2x2 B-block
    # A is 2x4, B is 4x6
    # Block structure:
    #   A rows 0-1, cols 0-1  times  B rows 0-1, cols 0-1  -> C rows 0-1, cols 0-1
    #   A rows 0-1, cols 0-1  times  B rows 0-1, cols 2-3  -> C rows 0-1, cols 2-3
    #   A rows 0-1, cols 0-1  times  B rows 0-1, cols 4-5  -> C rows 0-1, cols 4-5
    #   A rows 0-1, cols 2-3  times  B rows 2-3, cols 0-1  -> C rows 0-1, cols 0-1
    #   A rows 0-1, cols 2-3  times  B rows 2-3, cols 2-3  -> C rows 0-1, cols 2-3
    #   A rows 0-1, cols 2-3  times  B rows 2-3, cols 4-5  -> C rows 0-1, cols 4-5

    blocks = [
        # (a_row_start, a_col_start, b_row_start, b_col_start)
        (0, 0, 0, 0),  # Block 0
        (0, 0, 0, 2),  # Block 1
        (0, 0, 0, 4),  # Block 2
        (0, 2, 2, 0),  # Block 3
        (0, 2, 2, 2),  # Block 4
        (0, 2, 2, 4),  # Block 5
    ]

    col = 0
    for (a_rs, a_cs, b_rs, b_cs) in blocks:
        # Map local indices to global indices
        # A local: 0=(a_rs, a_cs), 1=(a_rs, a_cs+1), 2=(a_rs+1, a_cs), 3=(a_rs+1, a_cs+1)
        a_map = {
            0: a_rs * m + a_cs,
            1: a_rs * m + (a_cs + 1),
            2: (a_rs + 1) * m + a_cs,
            3: (a_rs + 1) * m + (a_cs + 1),
        }

        # B local: 0=(b_rs, b_cs), 1=(b_rs, b_cs+1), 2=(b_rs+1, b_cs), 3=(b_rs+1, b_cs+1)
        b_map = {
            0: b_rs * p + b_cs,
            1: b_rs * p + (b_cs + 1),
            2: (b_rs + 1) * p + b_cs,
            3: (b_rs + 1) * p + (b_cs + 1),
        }

        # C local: 0=(b_cs, a_rs), 1=(b_cs+1, a_rs), 2=(b_cs, a_rs+1), 3=(b_cs+1, a_rs+1)
        # W index: k*n + i where k is col in C, i is row in C
        w_map = {
            0: b_cs * n + a_rs,
            1: (b_cs + 1) * n + a_rs,
            2: b_cs * n + (a_rs + 1),
            3: (b_cs + 1) * n + (a_rs + 1),
        }

        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff

            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff

            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff

            col += 1

    return {"U": U, "V": V, "W": W}
