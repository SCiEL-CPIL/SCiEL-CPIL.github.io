"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_strassen_schoolbook(n, m, p):
    """Build the known rank-50 decomposition: 6 Strassen (2x2x2) + 2 schoolbook (2x2x1)."""
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    return U, V, W, col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition reproduces the all-ones tensor using matrix products."""
    r = U.shape[1]
    if r == 0:
        return False
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For small dimensions, direct computation is fast
    T_check = np.zeros((n, m, p), dtype=np.int64)
    for r_idx in range(r):
        # u_i = U[i, j, r_idx] for all i,j
        # v_jk = V[j, k, r_idx] for all j,k  
        # w_ki = W[k, i, r_idx] for all k,i
        u_col = U_t[:, :, r_idx]  # (n, m)
        v_col = V_t[:, :, r_idx]  # (m, p)
        w_col = W_t[:, :, r_idx]  # (p, n)
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        # Build as: for each i, u_col[i,:] is (m,), 
        #   for each j, u_col[i,j] * v_col[j,:] is (p,),
        #   for each k, multiply by w_col[k,i]
        # Use einsum for this single term
        T_check += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col, optimize=False)
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T_check, expected)

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using precomputed column tensors."""
    return _verify(U, V, W, n, m, p)

def _dual_constraint_analysis(U, V, W, n, m, p):
    """
    Analyze the dual tensor constraints (syzygies) to identify columns that are
    linearly dependent in the tensor sense.
    
    For a valid rank-r decomposition, each column r contributes a rank-1 tensor
    T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r].
    
    The dual constraints are: for any linear functional L on the tensor space,
    sum_r L(T_r) = L(T) where T is the all-ones tensor.
    
    We check if removing a set of columns and adjusting others still satisfies
    all dual constraints. This is equivalent to checking if the remaining
    columns still span the correct tensor.
    
    Returns: list of column indices that can potentially be removed (candidates).
    """
    r = U.shape[1]
    candidates = []
    
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Flatten each rank-1 term to a vector in R^(n*m*p)
    # term_r vector: for (i,j,k), value = U_t[i,j,r]*V_t[j,k,r]*W_t[k,i,r]
    # If any term is in the span of others, it can be removed.
    
    for r1 in range(r):
        for r2 in range(r1 + 1, r):
            pass
    
    return candidates

def _compute_column_tensors(U, V, W, n, m, p):
    """Precompute the tensor contribution of each column as a flat vector."""
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    col_tensors = np.zeros((r, n * m * p), dtype=np.int64)
    for r_idx in range(r):
        u_col = U_t[:, :, r_idx]
        v_col = V_t[:, :, r_idx]
        w_col = W_t[:, :, r_idx]
        t = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col, optimize=False)
        col_tensors[r_idx] = t.flatten()
    return col_tensors

def _greedy_set_cover(U, V, W, n, m, p, deadline):
    """
    Greedy search for lower-rank exact decomposition using precomputed column tensors.
    Uses linearity: removing column r means the remaining columns must sum to
    ones_vector - col_tensor[r].
    """
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    # --- Step 1: Remove zero columns (trivial reduction) ---
    zero_mask = (np.all(U == 0, axis=0) | np.all(V == 0, axis=0) | np.all(W == 0, axis=0))
    if zero_mask.any():
        keep = ~zero_mask
        U, V, W = U[:, keep], V[:, keep], W[:, keep]
        current_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank

    if time.time() > deadline - 10:
        return (best_U, best_V, best_W) if best_rank < 50 else None

    # Precompute column tensor contributions
    col_tensors = _compute_column_tensors(U, V, W, n, m, p)
    ones_vec = np.ones(n * m * p, dtype=np.int64)
    total_sum = col_tensors.sum(axis=0)
    
    # Verify initial
    if not np.array_equal(total_sum, ones_vec):
        return None

    # --- Step 2: Iterative single-column removal using precomputed tensors ---
    U_work, V_work, W_work = U.copy(), V.copy(), W.copy()
    rank_work = U_work.shape[1]
    col_tensors_work = col_tensors.copy()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r in range(rank_work):
            # Check if removing column r still gives correct sum
            new_sum = col_tensors_work.sum(axis=0) - col_tensors_work[r]
            if np.array_equal(new_sum, ones_vec):
                # Remove column r
                mask = np.ones(rank_work, dtype=bool)
                mask[r] = False
                U_work = U_work[:, mask]
                V_work = V_work[:, mask]
                W_work = W_work[:, mask]
                col_tensors_work = col_tensors_work[mask]
                rank_work = U_work.shape[1]
                improved = True
                best_U, best_V, best_W = U_work.copy(), V_work.copy(), W_work.copy()
                best_rank = rank_work
                break

    if best_rank < 50:
        return (best_U, best_V, best_W)

    # --- Step 3: Pairwise column elimination using precomputed tensors ---
    if time.time() < deadline - 3:
        U_try, V_try, W_try = best_U.copy(), best_V.copy(), best_W.copy()
        rank_try = U_try.shape[1]
        col_tensors_try = _compute_column_tensors(U_try, V_try, W_try, n, m, p)
        total_try = col_tensors_try.sum(axis=0)
        
        for r1 in range(rank_try):
            for r2 in range(r1 + 1, rank_try):
                if time.time() > deadline - 3:
                    break
                new_sum = total_try - col_tensors_try[r1] - col_tensors_try[r2]
                if np.array_equal(new_sum, ones_vec):
                    mask = np.ones(rank_try, dtype=bool)
                    mask[r1] = False
                    mask[r2] = False
                    U_try = U_try[:, mask]
                    V_try = V_try[:, mask]
                    W_try = W_try[:, mask]
                    col_tensors_try = col_tensors_try[mask]
                    total_try = col_tensors_try.sum(axis=0)
                    rank_try = U_try.shape[1]
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = rank_try
                    break
            if time.time() > deadline - 3:
                break
    
    if best_rank < 50:
        return (best_U, best_V, best_W)
    
    return None

def _search_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Search for lower-rank decompositions using precomputed column tensors.
    Tries removing 1, 2, or 3 columns at a time.
    """
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    col_tensors = _compute_column_tensors(U, V, W, n, m, p)
    ones_vec = np.ones(n * m * p, dtype=np.int64)
    total = col_tensors.sum(axis=0)
    
    if not np.array_equal(total, ones_vec):
        return None

    U_work, V_work, W_work = U.copy(), V.copy(), W.copy()
    rank_work = U_work.shape[1]
    ct_work = col_tensors.copy()
    
    # Try removing 1 column at a time
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        for r in range(rank_work):
            new_sum = ct_work.sum(axis=0) - ct_work[r]
            if np.array_equal(new_sum, ones_vec):
                mask = np.ones(rank_work, dtype=bool)
                mask[r] = False
                U_work = U_work[:, mask]
                V_work = V_work[:, mask]
                W_work = W_work[:, mask]
                ct_work = ct_work[mask]
                rank_work = U_work.shape[1]
                improved = True
                if rank_work < best_rank:
                    best_U, best_V, best_W = U_work.copy(), V_work.copy(), W_work.copy()
                    best_rank = rank_work
                break
    
    # Try removing 2 columns at a time
    if time.time() < deadline - 5:
        U_try, V_try, W_try = best_U.copy(), best_V.copy(), best_W.copy()
        rank_try = U_try.shape[1]
        ct_try = _compute_column_tensors(U_try, V_try, W_try, n, m, p)
        total_try = ct_try.sum(axis=0)
        
        for r1 in range(rank_try):
            for r2 in range(r1 + 1, rank_try):
                if time.time() > deadline - 5:
                    break
                new_sum = total_try - ct_try[r1] - ct_try[r2]
                if np.array_equal(new_sum, ones_vec):
                    mask = np.ones(rank_try, dtype=bool)
                    mask[r1] = False
                    mask[r2] = False
                    U_try = U_try[:, mask]
                    V_try = V_try[:, mask]
                    W_try = W_try[:, mask]
                    ct_try = ct_try[mask]
                    total_try = ct_try.sum(axis=0)
                    rank_try = U_try.shape[1]
                    if rank_try < best_rank:
                        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                        best_rank = rank_try
                    break
            if time.time() > deadline - 5:
                break
    
    # Try removing 3 columns at a time (only if rank is small enough)
    if time.time() < deadline - 3 and best_rank <= 30:
        U_try, V_try, W_try = best_U.copy(), best_V.copy(), best_W.copy()
        rank_try = U_try.shape[1]
        ct_try = _compute_column_tensors(U_try, V_try, W_try, n, m, p)
        total_try = ct_try.sum(axis=0)
        
        for r1 in range(rank_try):
            for r2 in range(r1 + 1, rank_try):
                for r3 in range(r2 + 1, rank_try):
                    if time.time() > deadline - 3:
                        break
                    new_sum = total_try - ct_try[r1] - ct_try[r2] - ct_try[r3]
                    if np.array_equal(new_sum, ones_vec):
                        mask = np.ones(rank_try, dtype=bool)
                        mask[r1] = False
                        mask[r2] = False
                        mask[r3] = False
                        U_try = U_try[:, mask]
                        V_try = V_try[:, mask]
                        W_try = W_try[:, mask]
                        ct_try = ct_try[mask]
                        total_try = ct_try.sum(axis=0)
                        rank_try = U_try.shape[1]
                        if rank_try < best_rank:
                            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                            best_rank = rank_try
                        break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break
    
    if best_rank < current_rank:
        return (best_U, best_V, best_W)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: evaluator kills at 150s

    n, m, p = DIMENSIONS

    # Build the known rank-50 decomposition
    U, V, W, col = _build_strassen_schoolbook(n, m, p)
    rank = col

    # Verify the rank-50 decomposition using fast method
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    # Attempt rank reduction via greedy set cover with local search
    if time.time() < deadline - 15:
        try:
            reduced = _greedy_set_cover(U, V, W, n, m, p, deadline)
            if reduced is not None:
                r_red = reduced[0].shape[1]
                if r_red < rank:
                    return {"U": reduced[0].tolist(), "V": reduced[1].tolist(), "W": reduced[2].tolist()}
        except Exception:
            pass

    # Try additional search for rank reduction
    if time.time() < deadline - 8:
        try:
            reduced2 = _search_rank_reduction(U, V, W, n, m, p, deadline)
            if reduced2 is not None:
                r_red2 = reduced2[0].shape[1]
                if r_red2 < rank:
                    return {"U": reduced2[0].tolist(), "V": reduced2[1].tolist(), "W": reduced2[2].tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
