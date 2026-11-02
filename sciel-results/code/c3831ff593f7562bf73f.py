import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # ---------- fast validity check ----------
    def check_validity_residual(U, V, W, tol=1e-6):
        R = U.shape[1]
        err = 0.0
        V_mat = V.reshape(m, p, R)
        U_mat = U.reshape(n, m, R)
        W_mat = W.reshape(p, n, R)
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

        # Part B: j=1,2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W

    # ---------- rank reduction via column sparsification ----------
    def try_reduce_rank(U, V, W, max_rank, time_limit=140.0):
        """
        Attempt to reduce rank by removing linearly dependent columns.
        Uses column space analysis to identify redundant columns.
        """
        current_rank = U.shape[1]
        if current_rank <= max_rank:
            return U, V, W

        # Try to find a subset of columns that spans the same tensor
        # We'll use a greedy approach: try removing columns one by one
        # and check if the decomposition is still valid.

        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()

        # Sort columns by "importance" - columns with larger norm are more important
        col_norms = np.linalg.norm(U_work, axis=0)
        col_indices = np.argsort(col_norms)[::-1]  # descending order

        # Try removing columns from least important to most important
        removed = set()
        for col in col_indices:
            if time.time() > time_limit:
                break
            if col in removed:
                continue
            if U_work.shape[1] - len(removed) <= max_rank:
                break

            # Try removing this column
            keep_cols = [c for c in range(U_work.shape[1]) if c != col and c not in removed]
            U_test = U_work[:, keep_cols]
            V_test = V_work[:, keep_cols]
            W_test = W_work[:, keep_cols]

            # Check if this removal preserves the decomposition
            err = check_validity_residual(U_test, V_test, W_test)
            if err < 1e-6:
                removed.add(col)
                U_work = U_test
                V_work = V_test
                W_work = W_test

        # Final check
        if U_work.shape[1] <= max_rank:
            err = check_validity_residual(U_work, V_work, W_work)
            if err < 1e-6:
                return U_work, V_work, W_work

        return U, V, W

    # ---------- local search via column transformations ----------
    def local_search(U, V, W, time_limit=140.0, max_iters=10000):
        """
        Perform local search by randomly permuting and linearly combining columns.
        Goal: find equivalent decompositions with sparser W entries or lower effective rank.
        """
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_score = float('inf')

        # Compute initial score (number of non-zero entries in W)
        def score(W_mat):
            # Count non-zero entries
            return np.count_nonzero(np.abs(W_mat) > 1e-10)

        best_score = score(best_W)

        R = U.shape[1]
        rng = random.Random(42)

        for it in range(max_iters):
            if time.time() > time_limit:
                break

            # Randomly select a pair of columns to mix
            if R < 2:
                break

            c1, c2 = rng.sample(range(R), 2)

            # Random mixing coefficients
            alpha = rng.uniform(-1.0, 1.0)
            beta = rng.uniform(-1.0, 1.0)

            # Create a transformation matrix
            T = np.eye(R)
            T[c1, c1] = 1.0
            T[c2, c2] = 1.0
            T[c1, c2] = alpha
            T[c2, c1] = beta

            # Apply transformation to U and V (right multiplication)
            # W gets transformed by T^{-1}
            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue

            U_new = U @ T
            V_new = V @ T
            W_new = W @ T_inv

            # Check validity
            err = check_validity_residual(U_new, V_new, W_new)
            if err < 1e-6:
                new_score = score(W_new)
                if new_score < best_score:
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    best_score = new_score

                    # Try to reduce rank further
                    U_red, V_red, W_red = try_reduce_rank(U_new, V_new, W_new, best_score, time_limit)
                    if U_red.shape[1] < best_U.shape[1]:
                        best_U, best_V, best_W = U_red, V_red, W_red
                        best_score = U_red.shape[1]

        return best_U, best_V, best_W

    # ---------- main ----------
    U, V, W = get_rank_55_solution()

    # Verify base solution
    err = check_validity_residual(U, V, W)
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank via local search
    if time.time() < deadline - 20.0:
        U, V, W = local_search(U, V, W, time_limit=deadline - 10.0)

    # Final verification
    err = check_validity_residual(U, V, W)
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
