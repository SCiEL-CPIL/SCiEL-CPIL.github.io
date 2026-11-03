"""Matrix multiplication 4,5,6 with Strassen-based rank reduction."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    """
    Rank-84 decomposition: Strassen(7) x schoolbook(4) for 4x4x4 + schoolbook for remainder.
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for (i, j), c in u_dict.items():
            u_vec[i * m + j] = c
        for (j, k), c in v_dict.items():
            v_vec[j * p + k] = c
        for (i, k), c in w_dict.items():
            w_vec[k * n + i] = c
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2: (a_coeffs, b_coeffs, c_coeffs)
    # A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Part 1: 4x4 @ 4x4 using Strassen outer + schoolbook inner (rank 28)
    # For each outer Strassen term s (7 terms), X_s and Y_s are 2x2 matrices.
    # X_s[pi,pj] = sum_b x_coeff[b] * A[(b//2)*2+pi, (b%2)*2+pj]
    # Y_s[pj,qk] = sum_b y_coeff[b] * B[(b//2)*2+pj, (b%2)*2+qk]
    # Schoolbook: Z_s[pi,qk] = sum_pj X_s[pi,pj] * Y_s[pj,qk]
    # Each product X_s[pi,pj] * Y_s[pj,qk] is ONE rank-1 term where:
    #   u has entries for all A elements in X_s[pi,pj]'s linear combo
    #   v has entries for all B elements in Y_s[pj,qk]'s linear combo
    #   w has entries for C elements that Z_s[pi,qk] contributes to

    for s in range(7):
        x_coeff = strassen_2x2[s][0]
        y_coeff = strassen_2x2[s][1]
        z_coeff = strassen_2x2[s][2]

        for pi in range(2):
            for pj in range(2):
                for qk in range(2):
                    # Build u_dict for X_s[pi,pj]
                    u_dict = {}
                    for b in range(4):
                        if x_coeff[b] != 0:
                            row = (b // 2) * 2 + pi
                            col = (b % 2) * 2 + pj
                            key = (row, col)
                            u_dict[key] = u_dict.get(key, 0) + x_coeff[b]

                    # Build v_dict for Y_s[pj,qk]
                    v_dict = {}
                    for b in range(4):
                        if y_coeff[b] != 0:
                            row = (b // 2) * 2 + pj
                            col = (b % 2) * 2 + qk
                            key = (row, col)
                            v_dict[key] = v_dict.get(key, 0) + y_coeff[b]

                    # Build w_dict: Z_s[pi,qk] contributes to C blocks
                    w_dict = {}
                    for cb in range(4):
                        if z_coeff[cb] != 0:
                            c_row = (cb // 2) * 2 + pi
                            c_col = (cb % 2) * 2 + qk
                            key = (c_row, c_col)
                            w_dict[key] = w_dict.get(key, 0) + z_coeff[cb]

                    if u_dict and v_dict and w_dict:
                        add_term(u_dict, v_dict, w_dict)

    # Part 2: 4x4 @ 4x2 schoolbook (rank 32)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})

    # Part 3: 4x1 @ 1x6 schoolbook (rank 24)
    for i in range(4):
        for k in range(6):
            add_term({(i, 4): 1}, {(4, k): 1}, {(i, k): 1})

    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0:
                W[idx][r] = val

    return {"U": U, "V": V, "W": W}
