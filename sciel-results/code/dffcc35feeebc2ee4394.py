"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 27 via exact per-output-entry linear solve."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for b in range(2):
            for a_blk in range(2):
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1

                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff

                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff

                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff

                    col += 1

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]

        # Check all diagonal entries = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(s - 1.0) > 1e-8:
                        return False

        # Check all off-diagonal entries = 0
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(s) > 1e-8:
                        return False
        return True

    def try_rank_27(U28, V28, W28, drop_col, time_budget):
        """
        Fix U,V to Strassen-28 with column drop_col removed.
        Solve for W exactly per output entry.
        Return (W, valid) or (None, False).
        """
        keep_cols = [c for c in range(28) if c != drop_col]
        U = U28[:, keep_cols]  # 8 x 27
        V = V28[:, keep_cols]  # 16 x 27

        W = np.zeros((p * n, 27))  # 8 x 27

        # For each output entry c = k*n + i, we need:
        #   sum_r U[a_r] * V[b_r] * W[c, r] = 1  for the diagonal (i,j,k)
        #   sum_r U[a_r] * V[b_r] * W[c, r] = 0  for all off-diagonal
        #
        # This is a linear system in W[c, :]:
        #   M_c @ W[c, :] = rhs_c
        # where M_c has one row per (a, b) pair, with M_c[ab, r] = U[a, r] * V[b, r]
        # and rhs_c[ab] = 1 if (a,b,c) is diagonal, 0 otherwise.
        #
        # There are 8*16 = 128 rows, 27 unknowns. Solve by least squares,
        # then check residual.

        for c in range(p * n):
            i2 = c // n
            k2 = c % n
            M = np.zeros((n * m * m * p, 27))
            rhs = np.zeros(n * m * m * p)
            idx = 0
            for a in range(n * m):
                i1 = a // m
                j1 = a % m
                for b in range(m * p):
                    j2 = b // p
                    k1 = b % p
                    M[idx, :] = U[a, :] * V[b, :]
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        rhs[idx] = 1.0
                    idx += 1

            # Solve least squares
            try:
                sol, residuals, rank_M, sv = np.linalg.lstsq(M, rhs, rcond=None)
            except:
                return None, False

            W[c, :] = sol

        # Check residual
        U_np = U
        V_np = V
        W_np = W
        max_res = 0.0
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        res = abs(s - 1.0)
                    else:
                        res = abs(s)
                    if res > max_res:
                        max_res = res

        if max_res < 1e-7:
            return W, True
        return None, False

    def refine_rank_27(U_init, V_init, W_init, time_budget):
        """
        ALS refinement with exact per-output-entry W solve at each step.
        U and V are updated by least squares; W is solved exactly per output entry.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        target_rank = 27

        for step in range(60):
            if time.time() > time_budget:
                break

            # Update U rows: for each a, solve min ||M_a @ U[a,:] - rhs_a||
            for a in range(n * m):
                i1 = a // m
                j1 = a % m
                M = np.zeros((m * p * p * n, target_rank))
                rhs = np.zeros(m * p * p * n)
                idx = 0
                for b in range(m * p):
                    j2 = b // p
                    k1 = b % p
                    for c in range(p * n):
                        i2 = c // n
                        k2 = c % n
                        M[idx, :] = V[b, :] * W[c, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        idx += 1
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Update V rows
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                M = np.zeros((n * m * p * n, target_rank))
                rhs = np.zeros(n * m * p * n)
                idx = 0
                for a in range(n * m):
                    i1 = a // m
                    j1 = a % m
                    for c in range(p * n):
                        i2 = c // n
                        k2 = c % n
                        M[idx, :] = U[a, :] * W[c, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        idx += 1
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Update W: exact per-output-entry solve
            for c in range(p * n):
                i2 = c // n
                k2 = c % n
                M = np.zeros((n * m * m * p, target_rank))
                rhs = np.zeros(n * m * m * p)
                idx = 0
                for a in range(n * m):
                    i1 = a // m
                    j1 = a % m
                    for b in range(m * p):
                        j2 = b // p
                        k1 = b % p
                        M[idx, :] = U[a, :] * V[b, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        idx += 1
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Check validity
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if validate(U_list, V_list, W_list):
                return U_list, V_list, W_list

        return None

    # Build baseline rank-28
    best_result = build_strassen_28()
    best_rank = 28

    U28 = np.array(best_result["U"], dtype=np.float64)
    V28 = np.array(best_result["V"], dtype=np.float64)
    W28 = np.array(best_result["W"], dtype=np.float64)

    found = False

    # Strategy 1: For each possible dropped column, try exact W solve
    for drop_col in range(28):
        if time.time() > deadline - 20:
            break
        W_sol, valid = try_rank_27(U28, V28, W28, drop_col, deadline - 20)
        if valid:
            keep_cols = [c for c in range(28) if c != drop_col]
            U_sol = U28[:, keep_cols].tolist()
            V_sol = V28[:, keep_cols].tolist()
            best_rank = 27
            best_result = {"U": U_sol, "V": V_sol, "W": W_sol.tolist()}
            found = True
            break

    # Strategy 2: ALS refinement from perturbed Strassen-28 with column dropped
    if not found and time.time() < deadline - 10:
        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            random.seed(42 + attempt * 1009)
            np.random.seed(42 + attempt * 1009)

            drop_col = random.randint(0, 27)
            keep_cols = [c for c in range(28) if c != drop_col]

            U0 = U28[:, keep_cols].copy()
            V0 = V28[:, keep_cols].copy()
            W0 = W28[:, keep_cols].copy()

            # Small perturbation to escape the exact-strassen manifold
            U0 += np.random.randn(n * m, 27) * 0.05
            V0 += np.random.randn(m * p, 27) * 0.05
            W0 += np.random.randn(p * n, 27) * 0.05

            result = refine_rank_27(U0, V0, W0, deadline - 10)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = 27
                best_result = {"U": U_sol, "V": V_sol, "W": W_sol}
                found = True
                break

    # Strategy 3: Try all drop columns with ALS refinement
    if not found and time.time() < deadline - 5:
        for drop_col in range(28):
            if time.time() > deadline - 5:
                break

            keep_cols = [c for c in range(28) if c != drop_col]
            U0 = U28[:, keep_cols].copy()
            V0 = V28[:, keep_cols].copy()
            W0 = W28[:, keep_cols].copy()

            result = refine_rank_27(U0, V0, W0, deadline - 5)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = 27
                best_result = {"U": U_sol, "V": V_sol, "W": W_sol}
                found = True
                break

    return best_result

# EVOLVE-BLOCK-END
