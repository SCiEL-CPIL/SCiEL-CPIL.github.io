import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_4x4x4_terms():
        """
        Returns 49 (u, v, w) triplets for 4x4 * 4x4 multiplication.
        Uses recursive Strassen: outer 7 terms x inner 7 terms = 49.
        """
        outer_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (A10+A11)B00 -> C10 - C11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = A00(B01-B11) -> C01 + C11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = A11(B10-B00) -> C00 + C10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (A00+A01)B11 -> -C00 + C01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (A10-A00)(B00+B01) -> C11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (A01-A11)(B10+B11) -> C00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_terms = [
            # M1 = (x00+x11)(y00+y11) -> z00 + z11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (x10+x11)y00 -> z10 - z11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = x00(y01-y11) -> z01 + z11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = x11(y10-y00) -> z00 + z10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (x00+x01)y11 -> -z00 + z01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (x10-x00)(y00+y01) -> z11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (x01-x11)(y10+y11) -> z00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices(block_idx, local_idx, dim, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                return row * p + col
            elif dim == 'C':
                return col * n + row

        result = []
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

                result.append((u, v, w))

        return result

    def strassen_4x4x3_tail():
        """
        Generate 4x4 * 4x3 tail (cols 4,5,6) using Strassen-4x4x4
        with column 3 of B zeroed out. This eliminates 14 terms
        (those whose V factor is entirely in the zeroed column),
        leaving 35 non-zero terms.
        """
        base_terms = strassen_4x4x4_terms()
        for (u, v, w) in base_terms:
            # Zero out column 3 of V (indices 3*p+0..3*p+3 = 21..24)
            for k in range(n):
                v[3 * p + k] = 0
            # Zero out column 3 of W (indices 3*n+0..3*n+3 = 12..15)
            for i in range(n):
                w[3 * n + i] = 0

            # Check if V is entirely zero (skip if so)
            if all(x == 0 for x in v):
                continue

            # Shift B and C column indices: col 4->0, 5->1, 6->2
            v_new = [0] * (m * p)
            w_new = [0] * (p * n)

            for j in range(m):
                for k in range(3):  # cols 0,1,2 (were 4,5,6)
                    v_new[j * p + k] = v[j * p + k]

            for k in range(3):  # cols 0,1,2
                for i in range(n):
                    w_new[k * n + i] = w[k * n + i]

            add_term(u, v_new, w_new)

    # Execute generation
    # Head: 4x4 * 4x4 (cols 0-3)
    head_terms = strassen_4x4x4_terms()
    for (u, v, w) in head_terms:
        add_term(u, v, w)

    # Tail: 4x4 * 4x3 (cols 4-6) via zeroed-column Strassen
    strassen_4x4x3_tail()

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
