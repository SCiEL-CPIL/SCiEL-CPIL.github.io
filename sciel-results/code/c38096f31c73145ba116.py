"""
Rank-minimal tensor decomposition for 4x5 @ 5x6 matrix multiplication.

The multiplication tensor T_{i,j,k} = 1 for all i in [0,4), j in [0,5), k in [0,6).
We seek U (20 x R), V (30 x R), W (24 x R) with integer/half-integer entries such that
sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all (i,j,k).

Strategy:
1. Use a fast NumPy-based error evaluation to enable many more local search steps.
2. Use a smarter initialization that starts from a valid rank-120 decomposition and
   then tries to merge columns (reduce rank) via local search.
3. Use a smooth surrogate (squared error) for the objective during local search,
   with exact integer verification at the end.
4. Apply a final polish pass with tighter tolerance.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = N * M * P  # 120

    # Precompute index arrays for fast tensor evaluation
    # For each (i,j,k), the row indices in U, V, W
    # U row: i*M + j
    # V row: j*P + k
    # W row: k*N + i
    idx_i = np.repeat(np.arange(N), M * P)
    idx_j = np.tile(np.repeat(np.arange(M), P), N)
    idx_k = np.tile(np.arange(P), N * M)

    U_rows = idx_i * M + idx_j
    V_rows = idx_j * P + idx_k
    W_rows = idx_k * N + idx_i

    def schoolbook():
        U = [[0] * target_rank for _ in range(N * M)]
        V = [[0] * target_rank for _ in range(M * P)]
        W = [[0] * target_rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, R):
        """Exact verification using integer arithmetic."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        # Compute all (i,j,k) values
        vals = np.zeros(N * M * P, dtype=np.int64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            # For each (i,j,k): u[i*M+j] * v[j*P+k] * w[k*N+i]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        return np.all(vals == 1)

    def compute_sqr_err(U_np, V_np, W_np, R):
        """Compute squared error using NumPy for speed."""
        vals = np.zeros(N * M * P, dtype=np.float64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        err = vals - 1.0
        return np.sum(err * err)

    def local_search_numpy(U, V, W, R, max_steps=50000):
        """
        Local search using NumPy for fast error evaluation.
        Tries to reduce squared error by modifying individual entries.
        """
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        cur_err = compute_sqr_err(U_np, V_np, W_np, R)
        if cur_err < 1e-12:
            # Verify exactly
            if verify(U, V, W, R):
                return U, V, W, 0

        step = 0
        no_improve_count = 0
        while step < max_steps and time.time() < deadline - 5:
            step += 1
            improved = False

            # Choose which matrix to modify
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat_np = U_np
                mat_list = U
                n_rows = N * M
            elif mat_choice < 0.66:
                mat_np = V_np
                mat_list = V
                n_rows = M * P
            else:
                mat_np = W_np
                mat_list = W
                n_rows = P * N

            row = random.randrange(n_rows)
            col = random.randrange(R)
            old_val = mat_np[row, col]

            # Try a few candidate new values
            candidates = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5]
            if old_val in candidates:
                candidates = [c for c in candidates if c != old_val]
            else:
                # Keep old as fallback
                pass

            best_candidate = None
            best_new_err = cur_err

            for new_val in candidates[:4]:  # Limit candidates for speed
                mat_np[row, col] = new_val
                new_err = compute_sqr_err(mat_np, V_np if mat_np is not V_np else U_np, W_np if mat_np is not W_np else U_np, R)
                # Need to be careful: we modified one of the matrices
                if mat_np is U_np:
                    new_err = compute_sqr_err(mat_np, V_np, W_np, R)
                elif mat_np is V_np:
                    new_err = compute_sqr_err(U_np, mat_np, W_np, R)
                else:
                    new_err = compute_sqr_err(U_np, V_np, mat_np, R)
                if new_err < best_new_err - 1e-15:
                    best_new_err = new_err
                    best_candidate = new_val
                mat_np[row, col] = old_val  # Restore

            if best_candidate is not None:
                mat_np[row, col] = best_candidate
                if mat_np is U_np:
                    mat_list[row][col] = best_candidate
                elif mat_np is V_np:
                    mat_list[row][col] = best_candidate
                else:
                    mat_list[row][col] = best_candidate
                cur_err = best_new_err
                improved = True
                no_improve_count = 0

                # Check for exact solution
                if cur_err < 1e-12:
                    # Convert to integers/half-integers for verification
                    U_int = [[round(U_np[i, r] * 2) / 2 for r in range(R)] for i in range(N * M)]
                    V_int = [[round(V_np[i, r] * 2) / 2 for r in range(R)] for i in range(M * P)]
                    W_int = [[round(W_np[i, r] * 2) / 2 for r in range(R)] for i in range(P * N)]
                    if verify(U_int, V_int, W_int, R):
                        return U_int, V_int, W_int, 0
            else:
                no_improve_count += 1
                if no_improve_count > 5000:
                    break

        # Convert back
        U_out = [[U_np[i, r] for r in range(R)] for i in range(N * M)]
        V_out = [[V_np[i, r] for r in range(R)] for i in range(M * P)]
        W_out = [[W_np[i, r] for r in range(R)] for i in range(P * N)]
        return U_out, V_out, W_out, int(cur_err)

    def try_reduce_rank_from_schoolbook(R_target):
        """
        Start from a valid rank-120 decomposition and try to reduce to R_target
        by merging columns and local search.
        """
        # Start with schoolbook
        U = [[0] * target_rank for _ in range(N * M)]
        V = [[0] * target_rank for _ in range(M * P)]
        W = [[0] * target_rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1

        # We want to reduce to R_target columns.
        # Strategy: randomly select R_target columns to keep, then add random
        # perturbations to allow the local search to find a valid decomposition.
        # Actually, a better strategy: initialize with R_target columns using
        # a structured pattern, then local search.

        U = [[0] * R_target for _ in range(N * M)]
        V = [[0] * R_target for _ in range(M * P)]
        W = [[0] * R_target for _ in range(P * N)]

        # Initialize with a structured pattern
        # Use the first R_target of the 120 schoolbook terms
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    if col >= R_target:
                        break
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
                if col >= R_target:
                    break
            if col >= R_target:
                break

        # Add random perturbations to all matrices
        for _ in range(R_target * 10):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randrange(N * M)
                r = random.randrange(R_target)
                U[idx][r] = random.choice([-1, 0, 1])
            elif mat == 'V':
                idx = random.randrange(M * P)
                r = random.randrange(R_target)
                V[idx][r] = random.choice([-1, 0, 1])
            else:
                idx = random.randrange(P * N)
                r = random.randrange(R_target)
                W[idx][r] = random.choice([-1, 0, 1])

        # Run local search
        U, V, W, err = local_search_numpy(U, V, W, R_target, max_steps=20000)

        if err == 0:
            return U, V, W

        return None

    # Try to find a decomposition with rank < 120
    # Start from just below 120 and work down
    for R in range(target_rank - 1, max(target_rank - 20, 100), -1):
        if time.time() > deadline - 20:
            break

        for restart in range(5):
            if time.time() > deadline - 10:
                break

            result = try_reduce_rank_from_schoolbook(R)
            if result is not None:
                U, V, W = result
                if verify(U, V, W, R):
                    return {"U": U, "V": V, "W": W}

    # If no improvement found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
