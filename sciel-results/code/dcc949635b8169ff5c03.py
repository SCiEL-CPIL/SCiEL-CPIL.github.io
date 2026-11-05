import numpy as np
import time

DIMENSIONS = [4, 5, 5]


def _fallback(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _tensor_from(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)


def _try_remove_columns(U, V, W, n, m, p, deadline, budget=10):
    """Greedy column removal. Returns (U,V,W,removed_count)."""
    r = U.shape[1]
    removed = 0
    for _ in range(budget):
        if time.time() > deadline:
            break
        if r == 0:
            break
        # Fast check: tensor contribution of each column
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        contribs = np.einsum('ijc,jkc,kic->ic', U3, V3, W3)
        # A column is removable if its contribution is all-zero (it adds nothing)
        removable = np.where(np.all(np.abs(contribs) < 1e-12, axis=1))[0]
        if len(removable) == 0:
            break
        # Remove the first removable column
        idx = removable[0]
        mask = np.ones(r, dtype=bool)
        mask[idx] = False
        U, V, W = U[:, mask], V[:, mask], W[:, mask]
        r = r - 1
        removed += 1
    return U, V, W, removed


def _try_merge_columns(U, V, W, n, m, p, deadline, max_pairs=20):
    """
    Try to merge pairs of columns that have identical (or proportional) support
    in a way that preserves the tensor. For each pair (a, b), check if the
    column vectors are scalar multiples. If U_a = c*U_b, V_a = c*V_b, W_a = c*W_b
    for some scalar c, then they are redundant and one can be removed.
    More generally, if the sum of two columns' contributions equals the contribution
    of a third, we could merge. But for now, just check for exact duplicates
    or scalar multiples which are the easiest to detect and remove.
    """
    r = U.shape[1]
    removed = 0
    # Check for scalar multiples
    for i in range(r):
        if time.time() > deadline:
            break
        for j in range(i + 1, r):
            # Check if col i is a scalar multiple of col j
            # Find a non-zero entry in col i of U
            nz_i = np.nonzero(U[:, i])[0]
            if len(nz_i) == 0:
                continue
            idx_i = nz_i[0]
            if U[idx_i, j] == 0:
                continue
            c_val = U[idx_i, i] / U[idx_i, j]
            # Check if all entries match
            if np.allclose(U[:, i], c_val * U[:, j], atol=1e-12) and \
               np.allclose(V[:, i], c_val * V[:, j], atol=1e-12) and \
               np.allclose(W[:, i], c_val * W[:, j], atol=1e-12):
                # Remove column j
                mask = np.ones(r, dtype=bool)
                mask[j] = False
                U, V, W = U[:, mask], V[:, mask], W[:, mask]
                r = r - 1
                removed += 1
                # Shift indices for next iteration
                j = i
                break
    return U, V, W, removed


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    ig = i_off + i_loc
                    jg = j_off + j_loc
                    kg = k_off + k_loc
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)

        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term([(gI(0,0), 1), (gI(1,1), 1)],
                 [(gJ(0,0), 1), (gJ(1,1), 1)],
                 [(gK(0,0), 1), (gK(1,1), 1)])
        # M2: (A10+A11)B00 -> C10 - C11
        add_term([(gI(1,0), 1), (gI(1,1), 1)],
                 [(gJ(0,0), 1)],
                 [(gK(0,1), 1), (gK(1,1), -1)])
        # M3: A00(B01-B11) -> C01 + C11
        add_term([(gI(0,0), 1)],
                 [(gJ(0,1), 1), (gJ(1,1), -1)],
                 [(gK(1,0), 1), (gK(1,1), 1)])
        # M4: A11(B10-B00) -> C00 + C10
        add_term([(gI(1,1), 1)],
                 [(gJ(1,0), 1), (gJ(0,0), -1)],
                 [(gK(0,0), 1), (gK(0,1), 1)])
        # M5: (A00+A01)B11 -> -C00 + C01
        add_term([(gI(0,0), 1), (gI(0,1), 1)],
                 [(gJ(1,1), 1)],
                 [(gK(0,0), -1), (gK(1,0), 1)])
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([(gI(1,0), 1), (gI(0,0), -1)],
                 [(gJ(0,0), 1), (gJ(0,1), 1)],
                 [(gK(1,1), 1)])
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([(gI(0,1), 1), (gI(1,1), -1)],
                 [(gJ(1,0), 1), (gJ(1,1), 1)],
                 [(gK(0,0), 1)])

    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)

    # Partition: n=4 as [2,2], m=5 as [2,3], p=5 as [2,3]
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]

    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)

    rank = len(U_cols)

    if rank == 0:
        U, V, W = _fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    # Verify correctness
    T = _tensor_from(U, V, W, n, m, p)
    if not np.allclose(T, 1.0, atol=1e-9):
        U, V, W = _fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank reduction: greedy column removal
    if time.time() < deadline:
        U, V, W, removed = _try_remove_columns(U, V, W, n, m, p, deadline, budget=10)
        # Re-verify after removal
        T = _tensor_from(U, V, W, n, m, p)
        if not np.allclose(T, 1.0, atol=1e-9):
            # Revert to pre-removal if removal broke correctness
            U_cols_orig = U_cols
            V_cols_orig = V_cols
            W_cols_orig = W_cols
            U = np.column_stack(U_cols_orig)
            V = np.column_stack(V_cols_orig)
            W = np.column_stack(W_cols_orig)

    # Try to merge columns (remove scalar multiples)
    if time.time() < deadline:
        U, V, W, removed2 = _try_merge_columns(U, V, W, n, m, p, deadline, max_pairs=50)
        T = _tensor_from(U, V, W, n, m, p)
        if not np.allclose(T, 1.0, atol=1e-9):
            # Revert if broken
            U_cols_orig = U_cols
            V_cols_orig = V_cols
            W_cols_orig = W_cols
            U = np.column_stack(U_cols_orig)
            V = np.column_stack(V_cols_orig)
            W = np.column_stack(W_cols_orig)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
