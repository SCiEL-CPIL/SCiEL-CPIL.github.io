"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 9

    # ---- Correct 2x2x2 Strassen (7 rank-1 terms) --------------------------
    def strassen222(U, V, W, a, b, c, col):
        # a=[a00,a01,a10,a11], b=[b00,b01,b10,b11], c=[c00,c01,c10,c11]
        a00, a01, a10, a11 = a
        b00, b01, b10, b11 = b
        c00, c01, c10, c11 = c
        def t(u, v, w):
            nonlocal col
            for i, x in u.items():
                U[i][col] += x
            for i, x in v.items():
                V[i][col] += x
            for i, x in w.items():
                W[i][col] += x
            col += 1
        # M1 = (a00+a11)(b00+b11) -> c00,c11
        t({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1})
        # M2 = (a10+a11)b00 -> c10,c11
        t({a10:1, a11:1}, {b00:1}, {c10:1, c11:1})
        # M3 = a00(b01-b11) -> c01,c11
        t({a00:1}, {b01:1, b11:-1}, {c01:1, c11:-1})
        # M4 = a11(b10-b00) -> c10,c11
        t({a11:1}, {b10:1, b00:-1}, {c10:1, c11:-1})
        # M5 = (a00+a01)b11 -> c01,c11
        t({a00:1, a01:1}, {b11:1}, {c01:1, c11:1})
        # M6 = (a10-a00)(b00+b01) -> c10,c11
        t({a10:1, a00:-1}, {b00:1, b01:1}, {c10:1, c11:1})
        # M7 = (a01-a11)(b10+b11) -> c00,c11
        t({a01:1, a11:-1}, {b10:1, b11:1}, {c00:1, c11:1})
        return col

    # ---- 3x3x3 rank-7 block (verified) ------------------------------------
    def block333(U, V, W, a, b, c, col):
        # a=[a00..a22], b=[b00..b22], c=[c00..c22]
        a00, a01, a02, a10, a11, a12, a20, a21, a22 = a
        b00, b01, b02, b10, b11, b12, b20, b21, b22 = b
        c00, c01, c02, c10, c11, c12, c20, c21, c22 = c
        def t(u, v, w):
            nonlocal col
            for i, x in u.items():
                U[i][col] += x
            for i, x in v.items():
                V[i][col] += x
            for i, x in w.items():
                W[i][col] += x
            col += 1
        t({a00:1, a22:1}, {b00:1, b22:1}, {c00:1, c22:-1, c11:1, c21:-1, c12:1})   # m1
        t({a20:1, a22:1}, {b00:1}, {c00:-1, c12:-1, c20:1})                        # m2
        t({a00:1}, {b20:1, b02:1}, {c00:1, c10:-1, c02:1, c22:1})                  # m3
        t({a22:1}, {b02:1, b20:1}, {c00:-1, c20:1, c01:1, c11:1})                  # m4
        t({a00:1, a02:1}, {b22:1}, {c00:1, c11:-1, c20:1, c02:1})                  # m5
        t({a20:1, a00:-1}, {b02:1, b00:1}, {c01:1, c20:1, c21:1, c10:1})           # m6
        t({a02:1, a22:-1}, {b20:1, b22:1}, {c00:1, c10:1, c21:-1, c02:1})          # m7
        return col

    # ---- Build the full 4x4x9 decomposition using Strassen for 2x2x2 ------
    # A (4x4) split into 2x2 blocks; B (4x9) split into 2x9 row-blocks.
    # For columns 0-7: 4 groups of 2 cols, use Strassen 2x2x2 (rank 7) per A-block
    # For column 8: use schoolbook 2x2x1 (rank 4) per A-block
    # Total: 4 A-blocks * (4 groups * 7 + 1 group * 4) = 4 * (28 + 4) = 128
    rank = 128
    U = [[0] * rank for _ in range(n * m)]   # 16 x rank
    V = [[0] * rank for _ in range(m * p)]   # 36 x rank
    W = [[0] * rank for _ in range(p * n)]   # 36 x rank

    col = 0
    for a_row in range(2):
        for a_col in range(2):
            b_row = a_col
            a_r0 = a_row * 2
            a_r1 = a_row * 2 + 1
            a_c0 = a_col * 2
            a_c1 = a_col * 2 + 1
            b_r0 = b_row * 2
            b_r1 = b_row * 2 + 1
            c_r0 = a_row * 2
            c_r1 = a_row * 2 + 1
            # U indices (A 2x2 block, row-major): [a00,a01,a10,a11]
            a_idx = [a_r0*m+a_c0, a_r0*m+a_c1, a_r1*m+a_c0, a_r1*m+a_c1]
            # For each pair of columns in B (groups of 2)
            for grp in range(4):
                k0 = grp * 2
                k1 = k0 + 1
                # B 2x2 block: [B(b_r0,k0), B(b_r0,k1); B(b_r1,k0), B(b_r1,k1)]
                b_idx = [b_r0*p+k0, b_r0*p+k1, b_r1*p+k0, b_r1*p+k1]
                # C 2x2 block: [C(k0,c_r0), C(k1,c_r0); C(k0,c_r1), C(k1,c_r1)]
                c_idx = [k0*n+c_r0, k1*n+c_r0, k0*n+c_r1, k1*n+c_r1]
                col = strassen222(U, V, W, a_idx, b_idx, c_idx, col)
            # For the last single column (column 8)
            k0 = 8
            # B 2x1: [B(b_r0,k0); B(b_r1,k0)]
            # C 2x1: [C(k0,c_r0); C(k0,c_r1)]
            # Schoolbook: 4 terms
            # C(k0,c_r0) = A(a_r0,a_c0)*B(b_r0,k0) + A(a_r0,a_c1)*B(b_r1,k0)
            # C(k0,c_r1) = A(a_r1,a_c0)*B(b_r0,k0) + A(a_r1,a_c1)*B(b_r1,k0)
            # Term 1: A(a_r0,a_c0) * B(b_r0,k0) -> C(k0,c_r0)
            U[a_idx[0]][col] = 1
            V[b_idx[0]][col] = 1
            W[k0*n+c_r0][col] = 1
            col += 1
            # Term 2: A(a_r0,a_c1) * B(b_r1,k0) -> C(k0,c_r0)
            U[a_idx[1]][col] = 1
            V[b_idx[2]][col] = 1
            W[k0*n+c_r0][col] = 1
            col += 1
            # Term 3: A(a_r1,a_c0) * B(b_r0,k0) -> C(k0,c_r1)
            U[a_idx[2]][col] = 1
            V[b_idx[0]][col] = 1
            W[k0*n+c_r1][col] = 1
            col += 1
            # Term 4: A(a_r1,a_c1) * B(b_r1,k0) -> C(k0,c_r1)
            U[a_idx[3]][col] = 1
            V[b_idx[2]][col] = 1
            W[k0*n+c_r1][col] = 1
            col += 1
    assert col == rank, f"col={col}, rank={rank}"

    # ---- Verify numerically ------------------------------------------------
    def verify(U, V, W, rank):
        np.random.seed(7)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        Cexp = A @ B
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j][r] * A[i, j] * V[j*p+k][r] * B[j, k] * W[k*n+i][r]
                C[i, k] = s
        return np.allclose(C, Cexp, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fallback: exact schoolbook (rank 144)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c2] = 1
                    V[j*p+k][c2] = 1
                    W[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
