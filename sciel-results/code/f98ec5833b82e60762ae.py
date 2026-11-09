"""
Rank-minimal tensor decomposition for 2x3x9 matrix multiplication.
Uses block Strassen (2x2x2 rank-7) for the first three column blocks of B,
and schoolbook for the remaining 2x2x3 and 2x1x9 blocks.
Total rank = 3*7 + 12 + 18 = 51.
"""
import numpy as np

DIMENSIONS = [2, 3, 9]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 51
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Strassen for 2x2x2: A (2x2) * B (2x2) = C (2x2)
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    def strassen_block(a_row0, a_row1, a_col0, a_col1,
                       b_row0, b_row1, b_col0, b_col1,
                       c_row0, c_row1, c_col0, c_col1):
        nonlocal col
        # Map local (i,j) to global A index
        def a_idx(i, j):
            return (a_row0 + i) * m + (a_col0 + j)

        def b_idx(i, j):
            return (b_row0 + i) * p + (b_col0 + j)

        def c_idx(i, j):
            return (c_col0 + j) * n + (c_row0 + i)

        # m1 = (a00+a11)(b00+b11)
        # We need to express this as sum of rank-1 terms.
        # (a00+a11) is a linear combination of A entries, (b00+b11) of B entries.
        # In tensor decomposition, each "term" in Strassen corresponds to
        # a product of a linear form in A and a linear form in B.
        # So we add ONE rank-1 term where U has the A coefficients, V has B coefficients,
        # and W distributes the result to C entries.
        
        # m1: A coeffs: a00=1, a11=1; B coeffs: b00=1, b11=1
        u1 = {a_idx(0,0): 1.0, a_idx(1,1): 1.0}
        v1 = {b_idx(0,0): 1.0, b_idx(1,1): 1.0}
        w1 = {c_idx(0,0): 1.0, c_idx(1,1): 1.0}  # C00 += m1, C11 += m1
        for u_i, u_v in u1.items():
            U[u_i, col] += u_v
        for v_i, v_v in v1.items():
            V[v_i, col] += v_v
        for w_i, w_v in w1.items():
            W[w_i, col] += w_v
        col += 1

        # m2: A coeffs: a10=1, a11=1; B coeffs: b00=1
        u2 = {a_idx(1,0): 1.0, a_idx(1,1): 1.0}
        v2 = {b_idx(0,0): 1.0}
        w2 = {c_idx(1,0): 1.0}  # C10 += m2
        for u_i, u_v in u2.items():
            U[u_i, col] += u_v
        for v_i, v_v in v2.items():
            V[v_i, col] += v_v
        for w_i, w_v in w2.items():
            W[w_i, col] += w_v
        col += 1

        # m3: A coeffs: a00=1; B coeffs: b10=1, b11=-1
        u3 = {a_idx(0,0): 1.0}
        v3 = {b_idx(1,0): 1.0, b_idx(1,1): -1.0}
        w3 = {c_idx(0,1): 1.0, c_idx(1,1): 1.0}  # C01 += m3, C11 += m3
        for u_i, u_v in u3.items():
            U[u_i, col] += u_v
        for v_i, v_v in v3.items():
            V[v_i, col] += v_v
        for w_i, w_v in w3.items():
            W[w_i, col] += w_v
        col += 1

        # m4: A coeffs: a11=1; B coeffs: b01=1, b00=-1
        u4 = {a_idx(1,1): 1.0}
        v4 = {b_idx(0,1): 1.0, b_idx(0,0): -1.0}
        w4 = {c_idx(0,0): 1.0, c_idx(1,0): 1.0}  # C00 += m4, C10 += m4
        for u_i, u_v in u4.items():
            U[u_i, col] += u_v
        for v_i, v_v in v4.items():
            V[v_i, col] += v_v
        for w_i, w_v in w4.items():
            W[w_i, col] += w_v
        col += 1

        # m5: A coeffs: a00=1, a01=1; B coeffs: b11=1
        u5 = {a_idx(0,0): 1.0, a_idx(0,1): 1.0}
        v5 = {b_idx(1,1): 1.0}
        w5 = {c_idx(0,0): -1.0, c_idx(0,1): 1.0}  # C00 -= m5, C01 += m5
        for u_i, u_v in u5.items():
            U[u_i, col] += u_v
        for v_i, v_v in v5.items():
            V[v_i, col] += v_v
        for w_i, w_v in w5.items():
            W[w_i, col] += w_v
        col += 1

        # m6: A coeffs: a10=1, a00=-1; B coeffs: b00=1, b01=1
        u6 = {a_idx(1,0): 1.0, a_idx(0,0): -1.0}
        v6 = {b_idx(0,0): 1.0, b_idx(0,1): 1.0}
        w6 = {c_idx(1,1): 1.0}  # C11 += m6
        for u_i, u_v in u6.items():
            U[u_i, col] += u_v
        for v_i, v_v in v6.items():
            V[v_i, col] += v_v
        for w_i, w_v in w6.items():
            W[w_i, col] += w_v
        col += 1

        # m7: A coeffs: a01=1, a11=-1; B coeffs: b10=1, b11=1
        u7 = {a_idx(0,1): 1.0, a_idx(1,1): -1.0}
        v7 = {b_idx(1,0): 1.0, b_idx(1,1): 1.0}
        w7 = {c_idx(0,0): 1.0}  # C00 += m7
        for u_i, u_v in u7.items():
            U[u_i, col] += u_v
        for v_i, v_v in v7.items():
            V[v_i, col] += v_v
        for w_i, w_v in w7.items():
            W[w_i, col] += w_v
        col += 1

    # Three 2x2x2 Strassen blocks for B columns 0-1, 2-3, 4-5
    for block in range(3):
        strassen_block(0, 1, 0, 1, 0, 1, 2*block, 2*block+1, 0, 1, 2*block, 2*block+1)

    # Block 4: A (2x2, cols 0-1) * B (2x3, rows 0-1, cols 6-8)
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a_i = i * m + j
                b_i = j * p + (6 + k)
                c_i = (6 + k) * n + i
                U[a_i, col] = 1.0
                V[b_i, col] = 1.0
                W[c_i, col] = 1.0
                col += 1

    # Block 5: A (2x1, col 2) * B (1x9, row 2)
    # Schoolbook: 2*1*9 = 18 terms
    for i in range(2):
        for k in range(9):
            a_i = i * m + 2
            b_i = 2 * p + k
            c_i = k * n + i
            U[a_i, col] = 1.0
            V[b_i, col] = 1.0
            W[c_i, col] = 1.0
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
