import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_rank_49(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if total != 1:
                    return False
    return True

def _strassen_rank_49(n, m, p):
    """
    Recursive Strassen 4x4: 7 outer block multiplications, each a 2x2 Strassen (7 scalar mults).
    Total rank = 7 * 7 = 49.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen 2x2 block formula (blocks are 2x2 matrices):
    outer_defs = [
        {
            'a': [(0,0,1),(1,1,1)],
            'b': [(0,0,1),(1,1,1)],
            'c': [(0,0,1),(1,1,1)],
        },
        {
            'a': [(1,0,1),(1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1),(1,1,-1)],
        },
        {
            'a': [(0,0,1)],
            'b': [(0,1,1),(1,1,-1)],
            'c': [(0,1,1),(1,1,1)],
        },
        {
            'a': [(1,1,1)],
            'b': [(1,0,1),(0,0,-1)],
            'c': [(0,0,1),(1,0,1)],
        },
        {
            'a': [(0,0,1),(0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,1,1),(0,0,-1)],
        },
        {
            'a': [(1,0,1),(0,0,-1)],
            'b': [(0,0,1),(0,1,1)],
            'c': [(1,1,1)],
        },
        {
            'a': [(0,1,1),(1,1,-1)],
            'b': [(1,0,1),(1,1,1)],
            'c': [(0,0,1)],
        },
    ]

    # Inner Strassen 2x2 multiplication: 7 scalar products
    # For 2x2 X * Y = Z, Strassen uses:
    # P0 = (X00+X11)*(Y00+Y11) -> Z00 += P0, Z11 += P0
    # P1 = (X10+X11)*Y00       -> Z10 += P1, Z11 -= P1
    # P2 = X00*(Y01-Y11)       -> Z01 += P2, Z11 += P2
    # P3 = X11*(Y10-Y00)       -> Z00 += P3, Z10 += P3
    # P4 = (X00+X01)*Y11       -> Z01 += P4, Z00 -= P4
    # P5 = (X10-X00)*(Y00+Y01) -> Z11 += P5
    # P6 = (X01-X11)*(Y10+Y11) -> Z00 += P6
    #
    # Each inner product t defines:
    #   x_contribs: list of (xi, xj, coeff) for X[xi, xj]
    #   y_contribs: list of (yj, yk, coeff) for Y[yj, yk]
    #   z_contribs: list of (zi, zk, coeff) for Z[zi, zk]
    inner_defs = [
        # P0
        {
            'x': [(0,0,1),(1,1,1)],
            'y': [(0,0,1),(1,1,1)],
            'z': [(0,0,1),(1,1,1)],
        },
        # P1
        {
            'x': [(1,0,1),(1,1,1)],
            'y': [(0,0,1)],
            'z': [(1,0,1),(1,1,-1)],
        },
        # P2
        {
            'x': [(0,0,1)],
            'y': [(0,1,1),(1,1,-1)],
            'z': [(0,1,1),(1,1,1)],
        },
        # P3
        {
            'x': [(1,1,1)],
            'y': [(1,0,1),(0,0,-1)],
            'z': [(0,0,1),(1,0,1)],
        },
        # P4
        {
            'x': [(0,0,1),(0,1,1)],
            'y': [(1,1,1)],
            'z': [(0,1,1),(0,0,-1)],
        },
        # P5
        {
            'x': [(1,0,1),(0,0,-1)],
            'y': [(0,0,1),(0,1,1)],
            'z': [(1,1,1)],
        },
        # P6
        {
            'x': [(0,1,1),(1,1,-1)],
            'y': [(1,0,1),(1,1,1)],
            'z': [(0,0,1)],
        },
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t

            # U contributions:
            # For each outer A-block (P_block, Q_block, sign_outer),
            #   the 2x2 sub-matrix A[P_block, Q_block] has entries A[2*P_block + xi, 2*Q_block + xj]
            # For each inner X contribution (xi, xj, sign_inner),
            #   U[global_row, global_col] += sign_outer * sign_inner
            for (Pb, Qb, so) in outer['a']:
                for (xi, xj, si) in inner['x']:
                    row = 2 * Pb + xi
                    col = 2 * Qb + xj
                    u_idx = row * m + col
                    U[u_idx][r] += so * si

            # V contributions:
            # For each outer B-block (P_block, Q_block, sign_outer),
            #   the 2x2 sub-matrix B[P_block, Q_block] has entries B[2*P_block + yj, 2*Q_block + yk]
            # For each inner Y contribution (yj, yk, sign_inner),
            #   V[global_row, global_col] += sign_outer * sign_inner
            for (Pb, Qb, so) in outer['b']:
                for (yj, yk, si) in inner['y']:
                    row = 2 * Pb + yj
                    col = 2 * Qb + yk
                    v_idx = row * p + col
                    V[v_idx][r] += so * si

            # W contributions:
            # For each outer C-block (P_block, Q_block, sign_outer),
            #   the 2x2 sub-matrix C[P_block, Q_block] has entries C[2*P_block + zi, 2*Q_block + zk]
            # For each inner Z contribution (zi, zk, sign_inner),
            #   W[global_col * n + global_row] += sign_outer * sign_inner
            for (Pb, Qb, so) in outer['c']:
                for (zi, zk, si) in inner['z']:
                    row = 2 * Pb + zi
                    col = 2 * Qb + zk
                    w_idx = col * n + row
                    W[w_idx][r] += so * si

    return U, V, W
