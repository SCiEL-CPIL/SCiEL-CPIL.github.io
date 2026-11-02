"""
Matrix multiplication 2x5x5 rank minimization.
Uses a Strassen-based rank-46 decomposition and then attempts rank
reductions by exploiting linear dependencies among the terms.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition inner dimension 5 = 2+2+1
    # Block (0,0): 2x2 x 2x2 -> Strassen
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # Block (0,2): 2x2 x 2x2 -> Strassen
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    # Block (0,4): 2x1 x 1x2 -> schoolbook (2*1*2 = 4 terms)
    schoolbook_block(0, 0, 0, 4, (2, 1), (1, 2), 0, 4)

    # Block (2,0): 2x2 x 2x2 -> Strassen
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    # Block (2,2): 2x2 x 2x2 -> Strassen
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    # Block (2,4): 2x1 x 1x2 -> schoolbook
    schoolbook_block(0, 2, 2, 4, (2, 1), (1, 2), 0, 4)

    # Block (4,0): 1x2 x 2x5 -> schoolbook (1*2*5 = 10 terms)
    schoolbook_block(0, 4, 4, 0, (1, 2), (2, 5), 0, 0)

    return terms


def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank


def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _contribution_matrix(terms):
    """
    Build the contribution matrix M where M[c, (i,j,k)] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    num_constraints = n * m * p  # 50

    M = np.zeros((r0, num_constraints))
    for c in range(r0):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx, val in terms[c][0]:
            u_dict[idx] = val
        for idx, val in terms[c][1]:
            v_dict[idx] = val
        for idx, val in terms[c][2]:
            w_dict[idx] = val

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    u_val = u_dict.get(idx_u, 0)
                    v_val = v_dict.get(idx_v, 0)
                    w_val = w_dict.get(idx_w, 0)
                    M[c, i * m * p + j * p + k] = u_val * v_val * w_val

    return M


def _find_null_space_reductions(terms, deadline):
    """
    Find linear dependencies among terms and use them to reduce rank.
    Iteratively apply null-space vectors to zero out coefficients.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    best_terms = terms
    best_rank = r0

    M = _contribution_matrix(terms)

    # Find null space of M^T: vectors v such that M^T @ v = 0
    # This means sum_c v[c] * M[c,:] = 0
    try:
        null_space = np.linalg.null_space(M.T)
    except:
        return terms

    if null_space is None or null_space.shape[1] == 0:
        return terms

    ns_dim = null_space.shape[1]

    # Iteratively apply reductions
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        if null_space is None or null_space.shape[1] == 0:
            break

        r0 = len(terms)
        found_reduction = False

        for ns in range(null_space.shape[1]):
            if time.time() > deadline - 5:
                break
            v = null_space[:, ns]

            # Find candidate t values that zero at least one coefficient
            # t = -1/v[c] for each c where v[c] != 0
            candidates = []
            for c in range(r0):
                if abs(v[c]) > 1e-10:
                    t_val = -1.0 / v[c]
                    candidates.append(t_val)

            best_t = None
            best_zero_count = 0
            for t_val in candidates:
                new_coeffs = 1.0 + t_val * v
                zero_count = sum(1 for c in range(r0) if abs(new_coeffs[c]) < 1e-9)
                if zero_count > best_zero_count:
                    best_zero_count = zero_count
                    best_t = t_val

            if best_t is not None and best_zero_count >= 1:
                new_coeffs = 1.0 + best_t * v
                new_terms = []
                for c in range(r0):
                    if abs(new_coeffs[c]) < 1e-9:
                        continue
                    scale = new_coeffs[c]
                    u_entries = [(idx, val * scale) for idx, val in terms[c][0]]
                    v_entries = [(idx, val) for idx, val in terms[c][1]]
                    w_entries = [(idx, val) for idx, val in terms[c][2]]
                    new_terms.append((u_entries, v_entries, w_entries))

                if len(new_terms) < best_rank:
                    U_test, V_test, W_test, _ = _terms_to_matrices(new_terms)
                    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                        best_terms = new_terms
                        best_rank = len(new_terms)
                        terms = new_terms
                        found_reduction = True
                        # Rebuild M and null space
                        M = _contribution_matrix(terms)
                        try:
                            null_space = np.linalg.null_space(M.T)
                        except:
                            break
                        break  # Restart iteration after a reduction

        if not found_reduction:
            # Try pairs of null-space vectors
            if null_space is not None and null_space.shape[1] >= 2:
                r0 = len(terms)
                for i in range(null_space.shape[1]):
                    for j in range(i + 1, null_space.shape[1]):
                        if time.time() > deadline - 5:
                            break
                        v = null_space[:, i]
                        w = null_space[:, j]

                        # Try to find b, c such that 1 + b*v + c*w has many zeros
                        # For each pair of indices (p1, p2), solve:
                        # b*v[p1] + c*w[p1] = -1
                        # b*v[p2] + c*w[p2] = -1
                        for p1 in range(r0):
                            for p2 in range(p1 + 1, r0):
                                if abs(v[p1]) < 1e-10 and abs(w[p1]) < 1e-10:
                                    continue
                                if abs(v[p2]) < 1e-10 and abs(w[p2]) < 1e-10:
                                    continue

                                A_mat = np.array([[v[p1], w[p1]], [v[p2], w[p2]]])
                                b_vec = np.array([-1.0, -1.0])
                                det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
                                if abs(det) < 1e-10:
                                    continue
                                try:
                                    bc = np.linalg.solve(A_mat, b_vec)
                                    b_val, c_val = bc
                                    new_coeffs = 1.0 + b_val * v + c_val * w
                                    zero_count = sum(1 for c in range(r0) if abs(new_coeffs[c]) < 1e-9)
                                    if zero_count >= 2:
                                        new_terms = []
                                        for c in range(r0):
                                            if abs(new_coeffs[c]) < 1e-9:
                                                continue
                                            scale = new_coeffs[c]
                                            u_entries = [(idx, val * scale) for idx, val in terms[c][0]]
                                            v_entries = [(idx, val) for idx, val in terms[c][1]]
                                            w_entries = [(idx, val) for idx, val in terms[c][2]]
                                            new_terms.append((u_entries, v_entries, w_entries))

                                        if len(new_terms) < best_rank:
                                            U_test, V_test, W_test, _ = _terms_to_matrices(new_terms)
                                            if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                                                best_terms = new_terms
                                                best_rank = len(new_terms)
                                                terms = new_terms
                                                found_reduction = True
                                                M = _contribution_matrix(terms)
                                                try:
                                                    null_space = np.linalg.null_space(M.T)
                                                except:
                                                    pass
                                                break
                                except:
                                    pass
                            if found_reduction:
                                break
                        if found_reduction:
                            break
                    if found_reduction:
                        break

        if not found_reduction:
            break

    return best_terms


def _try_alternative_tilings(deadline):
    """
    Try different partitions of the inner dimension to see if a lower
    base rank is achievable.
    """
    n, m, p = 2, 5, 5
    best_terms = None
    best_rank = 46

    # Try partition 5 = 1+1+1+1+1 (pure schoolbook) - rank 50, skip
    # Try partition 5 = 3+2
    # 3x3 blocks: use Strassen-like? For 2x3 x 3x3, rank is not simply 7
    # Let's just try the standard 2+2+1 which gives 46

    # Try partition 5 = 2+3
    terms_23 = []

    def add_term(u_entries, v_entries, w_entries):
        terms_23.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition: inner 5 = 2 + 3
    # A is 2x5, B is 5x5
    # j in {0,1}: 2x2 x 2x5 blocks
    #   B sub-block is 2x5: partition as 2x2 + 2x2 + 2x1
    #   Strassen for 2x2x2: 7 terms each, 2 of them = 14
    #   Schoolbook for 2x2x1: 4 terms
    #   Total for j={0,1}: 14 + 4 = 18? Wait, let me recount.
    #   Actually for j in {0,1}, B rows are {0,1}, B cols are {0,1,2,3,4}
    #   Partition B cols as {0,1}, {2,3}, {4}
    #   Block (j={0,1}, k={0,1}): 2x2 x 2x2 -> Strassen: 7
    #   Block (j={0,1}, k={2,3}): 2x2 x 2x2 -> Strassen: 7
    #   Block (j={0,1}, k={4}): 2x2 x 2x1 -> schoolbook: 4
    #   Subtotal: 18

    # j in {2,3,4}: 2x3 x 3x5 blocks
    #   Partition B rows as {2,3}, {4}
    #   Partition B cols as {0,1}, {2,3}, {4}
    #   Block (j={2,3}, k={0,1}): 2x2 x 2x2 -> Strassen: 7
    #   Block (j={2,3}, k={2,3}): 2x2 x 2x2 -> Strassen: 7
    #   Block (j={2,3}, k={4}): 2x2 x 2x1 -> schoolbook: 4
    #   Block (j={4}, k={0,1}): 2x1 x 1x2 -> schoolbook: 4
    #   Block (j={4}, k={2,3}): 2x1 x 1x2 -> schoolbook: 4
    #   Block (j={4}, k={4}): 2x1 x 1x1 -> schoolbook: 2
    #   Subtotal: 7+7+4+4+4+2 = 28

    # Total: 18 + 28 = 46. Same as before!

    # So 2+3 partition also gives 46. The 2+2+1 is just a specific case.

    # Let's stick with the 2+2+1 partition which we already have.
    return best_terms


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    terms = _build_strassen_decomposition()
    current_rank = len(terms)

    # Try null-space reductions iteratively
    if time.time() < deadline - 15:
        terms = _find_null_space_reductions(terms, deadline)
        current_rank = len(terms)

    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)

    # Verify
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback to original
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
