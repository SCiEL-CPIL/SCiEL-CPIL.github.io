import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        # 7 terms for 2x2x2 Strassen
        # A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[k_off:k_off+2, i_off:i_off+2]
        
        # M1 = (A00 + A11)(B01 - B11)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M2 = (A10 + A11) B00
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M3 = A00 (B10 - B11)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + i_off] = 1
        add_term(u, v, w)

        # M4 = A11 (B00 - B01)
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M5 = (A00 + A01) B11
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M6 = (A10 - A00) (B00 + B01)
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + i_off] = 1
        add_term(u, v, w)

        # M7 = (A01 - A11) (B10 + B11)
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        # 4x4x4 using 4x 2x2x2 Strassen? No, 49 terms.
        # Recursive Strassen: 7 * 7 = 49.
        # Outer Strassen on 4x4 blocks (each 2x2)
        # Inner Strassen on 2x2 blocks
        
        # This implementation matches the previous one's logic for 49 terms.
        # It iterates over 7 outer terms and 7 inner terms.
        
        def get_block_indices(block_idx, local_idx, dim):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                return row * p + col
            elif dim == 'C':
                return col * n + row

        # Outer Strassen terms for 2x2 blocks of A, B, C
        # A_blocks: 0:0,0:2, 1:0,1:2, 2:2,0:2, 3:2,1:2
        # Indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
        
        outer_terms = [
            ([(0, 1), (3, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),           # M2
            ([(0, 1)], [(2, 1), (3, -1)], [(0, 1), (1, 1)]),          # M3
            ([(3, 1)], [(0, 1), (1, -1)], [(2, 1), (3, 1)]),          # M4
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),           # M5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),  # M6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(2, 1), (3, 1)])   # M7
        ]
        
        # Inner Strassen terms for 2x2 blocks
        inner_terms = [
            ([(0, 1), (3, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
            ([(0, 1)], [(2, 1), (3, -1)], [(0, 1), (1, 1)]),
            ([(3, 1)], [(0, 1), (1, -1)], [(2, 1), (3, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(2, 1), (3, 1)])
        ]

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        # 4x4x2 using 4x 2x2x2 Strassen
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        # 4x4x1 using 16 terms
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
