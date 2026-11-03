import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    TIME_LIMIT = 120.0

    def time_up():
        return (time.time() - start_time) > TIME_LIMIT - 5.0

    def verify(U, V, W, rank):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])

        T_target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, T_target, atol=1e-9)

    def build_strassen():
        """Build the base rank-28 Strassen decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # Strassen steps for 2x2 matrix multiplication
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        strassen_steps = [
            # u: coefficients for A block (2x2), v: for B block (2x2), w: for C block (2x2)
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},  # M1
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},       # M2
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},        # M3
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},        # M4
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},        # M5
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},        # M6
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}         # M7
        ]

        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            """Add a 2x2 Strassen block at specified positions."""
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s

                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = l_row
                    global_j = a_col_start + l_col
                    global_idx = global_i * m + global_j
                    U[global_idx][col] += coeff

                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = b_row_start + l_row
                    global_j = b_col_start + l_col
                    global_idx = global_i * p + global_j
                    V[global_idx][col] += coeff

                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_k = c_col_start + l_col
                    global_i = l_row
                    global_idx = global_k * n + global_i
                    W[global_idx][col] += coeff

        # Four 2x2 blocks: A11*B11, A12*B21, A11*B12, A12*B22
        # A11: cols 0-1, A12: cols 2-3
        # B11: rows 0-1, cols 0-1; B12: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1; B22: rows 2-3, cols 2-3
        # C11: cols 0-1, C12: cols 2-3 (for row 0)
        # C21: cols 0-1, C22: cols 2-3 (for row 1)
        
        add_block(0, 0, 0, 0, 0)    # A11 * B11 -> C11
        add_block(0, 0, 2, 2, 7)    # A11 * B12 -> C12
        add_block(2, 2, 0, 0, 14)   # A12 * B21 -> C21
        add_block(2, 2, 2, 2, 21)   # A12 * B22 -> C22

        return U, V, W, rank

    def try_reduce_rank_fast(U, V, W, current_rank):
        """
        Try to reduce rank by finding linear dependencies using rank-revealing decomposition.
        Returns reduced (U, V, W, rank) or None if no reduction found.
        """
        if current_rank <= 1:
            return None

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Try to eliminate each column in order
        for elim_idx in range(current_rank - 1, -1, -1):
            if time_up():
                break

            other_indices = [j for j in range(current_rank) if j != elim_idx]
            
            # Check if column elim_idx is in the span of other columns
            U_target = U_arr[:, elim_idx]
            U_matrix = np.column_stack([U_arr[:, j] for j in other_indices])

            V_target = V_arr[:, elim_idx]
            V_matrix = np.column_stack([V_arr[:, j] for j in other_indices])

            W_target = W_arr[:, elim_idx]
            W_matrix = np.column_stack([W_arr[:, j] for j in other_indices])

            # Combine all constraints
            A_combined = np.vstack([U_matrix, V_matrix, W_matrix])
            b_combined = np.concatenate([U_target, V_target, W_target])

            # Check if b is in the column space of A
            # Use SVD for numerical stability
            try:
                U_svd, s, Vt = np.linalg.svd(A_combined, full_matrices=False)
                rank_A = np.sum(s > 1e-10)
                
                # Project b onto the column space
                if rank_A < len(s):
                    # If A is rank deficient, check if b is in the range
                    U_range = U_svd[:, :rank_A]
                    proj = U_range.T @ b_combined
                    recon = U_range @ proj
                    residual = np.linalg.norm(b_combined - recon)
                else:
                    # A is full column rank, solve least squares
                    coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_combined, b_combined, rcond=None)
                    recon = A_combined @ coeffs
                    residual = np.linalg.norm(b_combined - recon)
                
                if residual < 1e-9:
                    # Column is redundant, eliminate it
                    new_U = [U[i][:] for i in range(len(U))]
                    new_V = [V[i][:] for i in range(len(V))]
                    new_W = [W[i][:] for i in range(len(W))]

                    for new_idx in range(len(other_indices)):
                        j = other_indices[new_idx]
                        c = coeffs[new_idx] if 'coeffs' in locals() else 0
                        # Recompute coeffs properly
                        if 'coeffs' not in locals():
                            coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_combined, b_combined, rcond=None)
                        c = coeffs[new_idx]
                        if abs(c) > 1e-12:
                            for i in range(len(U)):
                                new_U[i][j] += c * U[i][elim_idx]
                                new_V[i][j] += c * V[i][elim_idx]
                                new_W[i][j] += c * W[i][elim_idx]

                    new_U = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_U]
                    new_V = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_V]
                    new_W = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_W]

                    return new_U, new_V, new_W, current_rank - 1
            except:
                pass

        return None

    # Start with Strassen
    U, V, W, rank = build_strassen()

    # Verify base case
    if not verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W, best_rank = U, V, W, rank

    # Try to reduce rank iteratively
    target_rank = 20
    while rank > target_rank and not time_up():
        result = try_reduce_rank_fast(U, V, W, rank)
        if result is None:
            break

        U, V, W, rank = result

        if verify(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            U, V, W, rank = best_U, best_V, best_W, best_rank
            break

    return {"U": best_U, "V": best_V, "W": best_W}
