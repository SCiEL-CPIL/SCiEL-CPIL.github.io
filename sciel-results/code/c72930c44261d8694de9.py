"""Matrix multiplication 3x4x8 - rank-minimal tensor decomposition.

Uses Strassen (rank 7) for 2x2x2 blocks and schoolbook for the remaining 1x2x8.
Total rank = 4*7 (Strassen for A[0:2,:] with B[0:2,:]) 
           + 4*7 (Strassen for A[0:2,:] with B[2:4,:])
           + 16 (schoolbook for A[2,:] with B[0:2,:])
           + 16 (schoolbook for A[2,:] with B[2:4,:])
           = 28 + 28 + 16 + 16 = 88
"""

DIMENSIONS = [3, 4, 8]


def construct():
    n, m, p = 3, 4, 8  # A: n x m, B: m x p, C: n x p
    rank = 88

    U = [[0] * rank for _ in range(n * m)]  # 12 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank

    col = [0]  # use list for mutability in nested function

    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        """Add one rank-1 term to the decomposition."""
        c = col[0]
        if c >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][c] += val
        for idx, val in v_coeffs.items():
            V[idx][c] += val
        for idx, val in w_coeffs.items():
            W[idx][c] += val
        col[0] = c + 1

    def add_strassen_2x2x2(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2,
                           out_r1, out_c1, out_r2, out_c2):
        """
        Add Strassen decomposition for multiplying a 2x2 sub-block of A
        by a 2x2 sub-block of B, contributing to a 2x2 sub-block of C.
        
        A sub-block indices: rows [a_r1, a_r2], cols [a_c1, a_c2]
        B sub-block indices: rows [b_r1, b_r2], cols [b_c1, b_c2]
        C output indices: rows [out_r1, out_r2], cols [out_c1, out_c2]
        
        W convention: W[k*n + i, r] where k is col of C, i is row of C, r is rank index
        So W index = col * n + row
        """
        # U indices: row * m + col
        u_a = a_r1 * m + a_c1  # A[a_r1][a_c1]
        u_b = a_r1 * m + a_c2  # A[a_r1][a_c2]
        u_c = a_r2 * m + a_c1  # A[a_r2][a_c1]
        u_d = a_r2 * m + a_c2  # A[a_r2][a_c2]

        # V indices: row * p + col
        v_e = b_r1 * p + b_c1  # B[b_r1][b_c1]
        v_f = b_r1 * p + b_c2  # B[b_r1][b_c2]
        v_g = b_r2 * p + b_c1  # B[b_r2][b_c1]
        v_h = b_r2 * p + b_c2  # B[b_r2][b_c2]

        # W indices: col * n + row
        w_c11 = out_c1 * n + out_r1  # C[out_r1][out_c1]
        w_c12 = out_c2 * n + out_r1  # C[out_r1][out_c2]
        w_c21 = out_c1 * n + out_r2  # C[out_r2][out_c1]
        w_c22 = out_c2 * n + out_r2  # C[out_r2][out_c2]

        # Strassen factors:
        # M1 = (a+d)(e+h) -> contributes to C11 and C22
        add_rank_unit(
            {u_a: 1, u_d: 1},
            {v_e: 1, v_h: 1},
            {w_c11: 1, w_c22: 1}
        )
        # M2 = (c+d)e -> contributes to C21 and C22
        add_rank_unit(
            {u_c: 1, u_d: 1},
            {v_e: 1},
            {w_c21: 1, w_c22: -1}
        )
        # M3 = a(f-h) -> contributes to C12 and C22
        add_rank_unit(
            {u_a: 1},
            {v_f: 1, v_h: -1},
            {w_c12: 1, w_c22: 1}
        )
        # M4 = d(g-e) -> contributes to C11 and C21
        add_rank_unit(
            {u_d: 1},
            {v_g: 1, v_e: -1},
            {w_c11: 1, w_c21: 1}
        )
        # M5 = (a+b)h -> contributes to C11 and C12
        add_rank_unit(
            {u_a: 1, u_b: 1},
            {v_h: 1},
            {w_c11: -1, w_c12: 1}
        )
        # M6 = (c-a)(e+f) -> contributes to C22
        add_rank_unit(
            {u_c: 1, u_a: -1},
            {v_e: 1, v_f: 1},
            {w_c22: 1}
        )
        # M7 = (b-d)(g+h) -> contributes to C11
        add_rank_unit(
            {u_b: 1, u_d: -1},
            {v_g: 1, v_h: 1},
            {w_c11: 1}
        )

    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # Split B into 4 blocks of 2 columns each
    for k_start in range(0, 8, 2):
        add_strassen_2x2x2(
            a_r1=0, a_c1=0, a_r2=1, a_c2=1,
            b_r1=0, b_c1=k_start, b_r2=1, b_c2=k_start + 1,
            out_r1=0, out_c1=k_start, out_r2=1, out_c2=k_start + 1
        )

    # Block 2: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :]
    for k_start in range(0, 8, 2):
        add_strassen_2x2x2(
            a_r1=0, a_c1=2, a_r2=1, a_c2=3,
            b_r1=2, b_c1=k_start, b_r2=3, b_c2=k_start + 1,
            out_r1=0, out_c1=k_start, out_r2=1, out_c2=k_start + 1
        )

    # Block 3: A[2:3, 0:2] * B[0:2, :] -> C[2:3, :]
    # Schoolbook: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    for k in range(8):
        # Term: A[2,0] * B[0,k]
        add_rank_unit(
            {2 * m + 0: 1},
            {0 * p + k: 1},
            {k * n + 2: 1}
        )
        # Term: A[2,1] * B[1,k]
        add_rank_unit(
            {2 * m + 1: 1},
            {1 * p + k: 1},
            {k * n + 2: 1}
        )

    # Block 4: A[2:3, 2:4] * B[2:4, :] -> C[2:3, :]
    # Schoolbook: C[2,k] = A[2,2]*B[2,k] + A[2,3]*B[3,k]
    for k in range(8):
        # Term: A[2,2] * B[2,k]
        add_rank_unit(
            {2 * m + 2: 1},
            {2 * p + k: 1},
            {k * n + 2: 1}
        )
        # Term: A[2,3] * B[3,k]
        add_rank_unit(
            {2 * m + 3: 1},
            {3 * p + k: 1},
            {k * n + 2: 1}
        )

    if col[0] != rank:
        raise Exception(f"Rank mismatch: used {col[0]}, expected {rank}")

    return {"U": U, "V": V, "W": W}
