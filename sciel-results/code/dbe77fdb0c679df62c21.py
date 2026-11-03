import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Rank 36 Strassen-based decomposition for 2x4x5
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]

        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val

            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val

            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val

            col_offset += 1

        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]

        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val

            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val

            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val

            col_offset += 1

        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)

    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that U, V, W form a valid tensor decomposition of the multiplication tensor."""
        r = U_mat.shape[1]
        # Check all (i,j,k) entries sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U_mat[i * m + j], V_mat[j * p + k]), W_mat[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        # Check all non-matching entries sum to 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip the matching entry
                    pass
        # More efficient: check all (a,b,c) triples where indices don't match
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            U_row = U_mat[a]
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                V_row = V_mat[b]
                UV = U_row * V_row
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.dot(UV, W_mat[c])
                    if abs(s) > tol:
                        return False
        return True

    if not verify_decomposition(U, V, W):
        return best_result

    # Tensor complement strategy: check for linear dependencies among the 36 rank-1 terms
    # The multiplication tensor T has entries T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
    # Each rank-1 term t_r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If sum_r alpha_r * t_r = 0, then the decomposition is not minimal
    # We can check this by flattening the tensor and looking at the span of the 36 vectors

    # Build the 36 rank-1 tensor vectors (flattened)
    tensor_dim = nm * mp * pn
    # Actually, we only need to check if the 36 vectors are linearly independent
    # in the space of tensors. But that's a huge space.
    # Instead, we can use the fact that if there's a linear dependency,
    # we can find it by checking the null space of a matrix.

    # Alternative approach: try to find if any column can be expressed as
    # a linear combination of others. This is equivalent to checking if
    # the rank of the matrix [U|V|W] (in some appropriate sense) is less than 36.

    # More practically: try dropping pairs of columns and solving for the rest.
    # This is more expensive but might find rank-34.

    # First, try the single-column drop (rank 35) as before
    def try_drop_column(drop_col):
        cols = [c for c in range(total_rank) if c != drop_col]
        U35 = U[:, cols]
        V35 = V[:, cols]

        num_vars = pn * 35
        num_eqs = nm * mp * pn  # This is too large, use the structure

        # Better: set up equations only for the entries that must be 1 or 0
        # For each (i,j,k) where i,j,k are valid, we need sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
        # For other combinations, we need sum_t ... = 0

        # Use the structured approach: only 40 equations for the "1" entries
        num_eqs = n * m * p
        A_sys = np.zeros((num_eqs, num_vars), dtype=np.float64)
        b_sys = np.ones(num_eqs, dtype=np.float64)

        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(35):
                        var_idx = (k * n + i) * 35 + t
                        A_sys[eq_idx, var_idx] = U35[i * m + j, t] * V35[j * p + k, t]
                    eq_idx += 1

        try:
            W35_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
        except np.linalg.LinAlgError:
            return None

        if len(residuals) > 0 and residuals[0] > 1e-8:
            return None

        W35 = W35_flat.reshape((pn, 35))

        if verify_decomposition(U35, V35, W35):
            U_round = np.round(U35 * 2) / 2
            V_round = np.round(V35 * 2) / 2
            W_round = np.round(W35 * 2) / 2

            if np.allclose(U_round, U35, atol=1e-9) and \
               np.allclose(V_round, V35, atol=1e-9) and \
               np.allclose(W_round, W35, atol=1e-9):
                if verify_decomposition(U_round, V_round, W_round):
                    return (U_round, V_round, W_round)

        return None

    # Try dropping each column
    for c in range(total_rank):
        if time.time() > deadline - 15:
            break
        result = try_drop_column(c)
        if result is not None:
            U35, V35, W35 = result
            U_list = [[float(U35[i, r]) for r in range(35)] for i in range(nm)]
            V_list = [[float(V35[i, r]) for r in range(35)] for i in range(mp)]
            W_list = [[float(W35[i, r]) for r in range(35)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Try dropping pairs of columns (rank 34)
    # This is more expensive, so we do it only if we have time
    if time.time() < deadline - 30:
        def try_drop_two_columns(drop_col1, drop_col2):
            cols = [c for c in range(total_rank) if c != drop_col1 and c != drop_col2]
            U34 = U[:, cols]
            V34 = V[:, cols]

            num_vars = pn * 34
            num_eqs = n * m * p
            A_sys = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b_sys = np.ones(num_eqs, dtype=np.float64)

            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(34):
                            var_idx = (k * n + i) * 34 + t
                            A_sys[eq_idx, var_idx] = U34[i * m + j, t] * V34[j * p + k, t]
                        eq_idx += 1

            try:
                W34_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            except np.linalg.LinAlgError:
                return None

            if len(residuals) > 0 and residuals[0] > 1e-8:
                return None

            W34 = W34_flat.reshape((pn, 34))

            if verify_decomposition(U34, V34, W34):
                U_round = np.round(U34 * 2) / 2
                V_round = np.round(V34 * 2) / 2
                W_round = np.round(W34 * 2) / 2

                if np.allclose(U_round, U34, atol=1e-9) and \
                   np.allclose(V_round, V34, atol=1e-9) and \
                   np.allclose(W_round, W34, atol=1e-9):
                    if verify_decomposition(U_round, V_round, W_round):
                        return (U_round, V_round, W_round)

            return None

        # Try all pairs
        for c1 in range(total_rank):
            if time.time() > deadline - 25:
                break
            for c2 in range(c1 + 1, total_rank):
                if time.time() > deadline - 25:
                    break
                result = try_drop_two_columns(c1, c2)
                if result is not None:
                    U34, V34, W34 = result
                    U_list = [[float(U34[i, r]) for r in range(34)] for i in range(nm)]
                    V_list = [[float(V34[i, r]) for r in range(34)] for i in range(mp)]
                    W_list = [[float(W34[i, r]) for r in range(34)] for i in range(pn)]
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Return rank-36 solution
    U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
    V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
    W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
