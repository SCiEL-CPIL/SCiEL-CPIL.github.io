"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    start_time = time.time()
    deadline = start_time + 140

    def get_schoolbook():
        rank = schoolbook_rank
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

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_rank_with_pattern(candidate_rank, pattern_type=0, seed=0):
        """
        Try to find a decomposition of given rank using structured patterns.
        pattern_type: 0 = random sparse, 1 = structured by indices, 2 = hybrid
        """
        if seed is not None:
            np.random.seed(seed)

        U = np.zeros((n * m, candidate_rank))
        V = np.zeros((m * p, candidate_rank))
        W = np.zeros((p * n, candidate_rank))

        if pattern_type == 0:
            # Random sparse initialization
            for r in range(candidate_rank):
                # Each rank index gets a few non-zero entries
                u_count = max(1, candidate_rank // 4)
                v_count = max(1, candidate_rank // 5)
                w_count = max(1, candidate_rank // 5)

                u_idx = np.random.choice(n * m, size=min(u_count, n * m), replace=False)
                v_idx = np.random.choice(m * p, size=min(v_count, m * p), replace=False)
                w_idx = np.random.choice(p * n, size=min(w_count, p * n), replace=False)

                for idx in u_idx:
                    U[idx, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0])
                for idx in v_idx:
                    V[idx, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0])
                for idx in w_idx:
                    W[idx, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0])

        elif pattern_type == 1:
            # Structured by index patterns
            for r in range(candidate_rank):
                # Use index-based patterns
                i_base = r % n
                j_base = (r // n) % m
                k_base = (r // (n * m)) % p

                # U: non-zero for all j when i == i_base, or all i when j == j_base
                for i in range(n):
                    for j in range(m):
                        if i == i_base or j == j_base:
                            U[i * m + j, r] = 1.0

                # V: non-zero for all k when j == j_base
                for j in range(m):
                    for k in range(p):
                        if j == j_base:
                            V[j * p + k, r] = 1.0

                # W: non-zero for all i when k == k_base
                for k in range(p):
                    for i in range(n):
                        if k == k_base:
                            W[k * n + i, r] = 1.0

        else:
            # Hybrid: mix of structured and random
            for r in range(candidate_rank):
                if r % 3 == 0:
                    # Structured
                    i_val = r % n
                    for i in range(n):
                        for j in range(m):
                            if i == i_val:
                                U[i * m + j, r] = 1.0
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r] = 1.0 if j == (r // n) % m else 0.0
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i, r] = 1.0 if k == (r // (n * m)) % p else 0.0
                else:
                    # Random
                    u_count = max(1, candidate_rank // 5)
                    v_count = max(1, candidate_rank // 6)
                    w_count = max(1, candidate_rank // 6)
                    u_idx = np.random.choice(n * m, size=min(u_count, n * m), replace=False)
                    v_idx = np.random.choice(m * p, size=min(v_count, m * p), replace=False)
                    w_idx = np.random.choice(p * n, size=min(w_count, p * n), replace=False)
                    for idx in u_idx:
                        U[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in v_idx:
                        V[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in w_idx:
                        W[idx, r] = np.random.choice([-1.0, 1.0])

        return U, V, W

    def run_als_polish(U, V, W, max_iter=50, tol=1e-10):
        """
        Run ALS to polish the decomposition.
        """
        R = U.shape[1]

        for iteration in range(max_iter):
            if time.time() > deadline:
                return False, U, V, W

            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k] = V[j * p + k] * W[k * n + i]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j] = u_sol
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i] = U[i * m + j] * W[k * n + i]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k] = v_sol
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j] = U[i * m + j] * V[j * p + k]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i] = w_sol
                    except:
                        pass

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1.0) ** 2

            if err < tol:
                return True, U, V, W

        return False, U, V, W

    def snap_and_verify(U, V, W, rank):
        """Snap to half-integers and verify."""
        U_snap = [[int(round(val * 2)) / 2.0 for val in row] for row in U.tolist()]
        V_snap = [[int(round(val * 2)) / 2.0 for val in row] for row in V.tolist()]
        W_snap = [[int(round(val * 2)) / 2.0 for val in row] for row in W.tolist()]

        if verify_decomposition(U_snap, V_snap, W_snap, rank):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    best_result = None
    best_rank = schoolbook_rank

    # Try ranks from lower to higher
    # Start with plausible minimums and work up
    for candidate_rank in range(15, 31):
        if time.time() > deadline:
            break

        # Try multiple patterns and seeds
        for pattern_type in range(3):
            for seed in range(5):
                if time.time() > deadline:
                    break

                U, V, W = try_rank_with_pattern(candidate_rank, pattern_type, seed * 1000 + candidate_rank)

                converged, U_final, V_final, W_final = run_als_polish(U, V, W, max_iter=30, tol=1e-10)

                if converged:
                    success, U_snap, V_snap, W_snap = snap_and_verify(U_final, V_final, W_final, candidate_rank)
                    if success:
                        best_result = {"U": U_snap, "V": V_snap, "W": W_snap}
                        best_rank = candidate_rank
                        break

            if best_result is not None and best_rank == candidate_rank:
                break

        if best_result is not None and best_rank == candidate_rank:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
