import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Define the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    def verify_fast(U_np, V_np, W_np, rank):
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1) > 1e-9:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        return verify_fast(U_np, V_np, W_np, rank)

    # Strassen-based rank 28 decomposition for 2x4x4
    def strassen_2x4x4():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        # Strassen terms for 2x2x2:
        # M1 = (X11+X22)(Y11+Y22)
        # M2 = (X21+X22)Y11
        # M3 = X11(Y12-Y22)
        # M4 = X22(Y21-Y11)
        # M5 = (X11+X12)Y22
        # M6 = (X21-X11)(Y11+Y12)
        # M7 = (X12-X22)(Y21+Y22)
        
        # Coefficients for U (X), V (Y), W (Z)
        # Format: list of (local_row, local_col, coeff)
        strassen_terms = [
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,0,1), (1,0,1)]),
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,1,-1)]),
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
        ]

        # 4 blocks:
        # P1: A11 * B11 -> C11
        # A11: i=0,1 j=0,1
        # B11: j=0,1 k=0,1
        # C11: k=0,1 i=0,1
        # P2: A11 * B12 -> C12
        # A11: i=0,1 j=0,1
        # B12: j=0,1 k=2,3
        # C12: k=2,3 i=0,1
        # P3: A12 * B21 -> C11
        # A12: i=0,1 j=2,3
        # B21: j=2,3 k=0,1
        # C11: k=0,1 i=0,1
        # P4: A12 * B22 -> C12
        # A12: i=0,1 j=2,3
        # B22: j=2,3 k=2,3
        # C12: k=2,3 i=0,1

        blocks = [
            # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
            # a: A[i, j] -> i is row, j is col
            # b: B[j, k] -> j is row, k is col
            # c: C[k, i] -> k is row, i is col (in W matrix index k*n+i)
            
            # P1
            (0, 0, 0, 0, 0, 0),
            # P2
            (0, 0, 0, 2, 2, 0),
            # P3
            (0, 2, 2, 0, 0, 0),
            # P4
            (0, 2, 2, 2, 2, 0),
        ]

        col = 0
        for a_ro, a_co, b_ro, b_co, c_ro, c_co in blocks:
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for (ar, ac, coeff_u) in u_coeffs:
                    global_ar = a_ro + ar
                    global_ac = a_co + ac
                    U[global_ar * m + global_ac][col] += coeff_u
                for (br, bc, coeff_v) in v_coeffs:
                    global_br = b_ro + br
                    global_bc = b_co + bc
                    V[global_br * p + global_bc][col] += coeff_v
                for (cr, cc, coeff_w) in w_coeffs:
                    global_cr = c_ro + cr
                    global_cc = c_co + cc
                    # W index is k*n + i
                    # In our block mapping:
                    # c_ro is k offset, c_co is i offset
                    # So global k = c_ro + cr, global i = c_co + cc
                    W[global_cr * n + global_cc][col] += coeff_w
                col += 1

        return {"U": U, "V": V, "W": W}

    if time.time() < deadline - 10:
        strassen = strassen_2x4x4()
        if verify(strassen["U"], strassen["V"], strassen["W"], 28):
            best = strassen
            best_rank = 28

    return best
# EVOLVE-BLOCK-END
