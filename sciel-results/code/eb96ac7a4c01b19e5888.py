import numpy as np
import time
from fractions import Fraction

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28

    # Correct Strassen 2x2 definitions
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (row, col, coeff) for A (2x2)
    # v_coeffs: list of (row, col, coeff) for B (2x2)
    # w_coeffs: list of (row, col, coeff) for C (2x2)

    strassen_2x2 = [
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    rank_444 = 49
    rank_443 = 4 * (7 + 2 * 2)  # 4 subproblems, each: 7 Strassen + 4 schoolbook = 11
    total_rank = rank_444 + rank_443

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # --- Part 1: 4x4 x 4x4 using Strassen-of-Strassen (rank 49) ---
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx

            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col

            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col

            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col

    # --- Part 2: 4x4 x 4x3 (cols 4-6 of B) ---
    offset = rank_444
    col = 0

    subproblems = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2),
    ]

    for (ar, ac, br, cr) in subproblems:
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_2x2:
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)

            for (lr, lc, coeff) in u_coeffs:
                i = ar + lr
                j = ac + lc
                a_idx = i * m + j
                U_col[a_idx] += coeff

            for (lr, lc, coeff) in v_coeffs:
                i = br + lr
                j = 4 + lc
                b_idx = i * p + j
                V_col[b_idx] += coeff

            for (lr, lc, coeff) in w_coeffs:
                i = cr + lr
                k = 4 + lc
                c_idx = k * n + i
                W_col[c_idx] += coeff

            U[:, offset + col] = U_col
            V[:, offset + col] = V_col
            W[:, offset + col] = W_col
            col += 1

        for lr in range(2):
            for j in range(2):
                i = ar + lr
                j_a = ac + j
                j_b = br + j
                k = 6

                a_idx = i * m + j_a
                b_idx = j_b * p + k
                c_idx = k * n + i

                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1

    # Fast validation using numpy
    if time.time() - start_time < DEADLINE - 15:
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > 1e-10:
                        raise ValueError(f"Validation failed for target ({a},{b},{c}): {val}")

        # Spot-check non-target entries (reduced count for speed)
        import random
        rng = random.Random(42)
        non_target_samples = set()
        for _ in range(500):
            a = rng.randint(0, nm - 1)
            b = rng.randint(0, mp - 1)
            c = rng.randint(0, pn - 1)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                non_target_samples.add((a, b, c))

        for (a, b, c) in non_target_samples:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-10:
                raise ValueError(f"Validation failed for non-target ({a},{b},{c}): {val}")

    # Try to reduce rank by removing linearly dependent columns
    if time.time() - start_time < DEADLINE - 10:
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()

        removed = set()
        for r1 in range(total_rank):
            if r1 in removed:
                continue
            for r2 in range(r1 + 1, total_rank):
                if r2 in removed:
                    continue
                U1 = U_work[:, r1]
                U2 = U_work[:, r2]
                V1 = V_work[:, r1]
                V2 = V_work[:, r2]
                W1 = W_work[:, r1]
                W2 = W_work[:, r2]

                nonzero_u1 = np.nonzero(U1)[0]
                if len(nonzero_u1) == 0:
                    continue
                idx = nonzero_u1[0]
                if U1[idx] == 0:
                    continue
                alpha = U2[idx] / U1[idx]

                if (np.allclose(U1 * alpha, U2, atol=1e-12) and
                    np.allclose(V1 * alpha, V2, atol=1e-12) and
                    np.allclose(W1 * alpha, W2, atol=1e-12)):
                    removed.add(r2)
                    break

        if removed:
            keep = [r for r in range(total_rank) if r not in removed]
            U_final = U_work[:, keep]
            V_final = V_work[:, keep]
            W_final = W_work[:, keep]
            new_rank = len(keep)

            # Fast validation of reduced decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_final[a, :] * V_final[b, :] * W_final[c, :])
                        if abs(val - 1) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                U = U_final
                V = V_final
                W = W_final
                total_rank = new_rank

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
