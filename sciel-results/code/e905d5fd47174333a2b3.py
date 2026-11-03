"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]

    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]

    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)

    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]

    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)

    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]

    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)

    # Verify the decomposition
    def verify_decomp(U_list, V_list, W_list, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rnk):
                        total += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if total != 1:
                        return False
        return True

    if not verify_decomp(U, V, W, rank):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (n*m*p, rank) = (48, 42)

    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result

    def verify_float(U_f, V_f, W_f, rnk, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rnk):
                        total += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Strategy 1: Systematic null space elimination with multiple orderings
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))

        if effective_rank < rank:
            # Try multiple elimination strategies
            for strategy in range(5):
                if time.time() > deadline - 20:
                    break

                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()

                if strategy == 0:
                    # Greedy: pick column with largest null vector coefficient
                    pass
                elif strategy == 1:
                    # Reverse order
                    pass
                elif strategy == 2:
                    # Random order
                    np.random.seed(strategy * 100 + 42)
                    pass
                elif strategy == 3:
                    # Sparsest column first
                    pass
                elif strategy == 4:
                    # Most central column
                    pass

                for iteration in range(rank - effective_rank + 5):
                    if time.time() > deadline - 20:
                        break

                    remaining = [c for c in range(rank) if c not in eliminated]
                    if len(remaining) <= 1:
                        break

                    M_sub = M[:, remaining]

                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break

                    tol_sub = 1e-9
                    eff_rank_sub = int(np.sum(s_sub > tol_sub))

                    if eff_rank_sub >= len(remaining):
                        break

                    null_sub = Vt_sub[eff_rank_sub:, :]

                    best_col = None
                    best_score = -1
                    best_null_idx = 0

                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            if strategy == 0:
                                score = abs(nv[local_c])
                            elif strategy == 1:
                                score = abs(nv[local_c]) if local_c == len(remaining)-1 else -1
                            elif strategy == 2:
                                score = abs(nv[local_c]) + np.random.random() * 0.05
                            elif strategy == 3:
                                # Prefer columns with sparse U,V,W patterns
                                sparsity = 1.0 / (np.sum(np.abs(new_U[:, global_c]) > 1e-10) +
                                                  np.sum(np.abs(new_V[:, global_c]) > 1e-10) +
                                                  np.sum(np.abs(new_W[:, global_c]) > 1e-10) + 1)
                                score = abs(nv[local_c]) * sparsity
                            else:
                                score = abs(nv[local_c]) * (1.0 / (1.0 + abs(local_c - len(remaining)/2.0)))

                            if score > best_score and abs(nv[local_c]) > 1e-8:
                                best_score = score
                                best_col = global_c
                                best_null_idx = nv_idx

                    if best_col is None:
                        break

                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]

                    if abs(pivot_val) < 1e-10:
                        continue

                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]

                    eliminated.add(best_col)

                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]

                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass

    # Strategy 2: Exact rational null space search
    # The tensor entries are integers, so we can find exact rational dependencies
    try:
        if time.time() < deadline - 30:
            # Use exact integer arithmetic to find the rank
            # M has integer entries (since U,V,W are integers)
            M_int = M.astype(np.int64)

            # Find exact rank using Gauss-Jordan over rationals
            # Work with the integer matrix
            def exact_rank(mat):
                """Compute exact rank of integer matrix using fraction-free elimination."""
                if mat.size == 0:
                    return 0
                rows, cols = mat.shape
                mat_f = mat.astype(object)
                rank = 0
                for col in range(min(rows, cols)):
                    # Find pivot
                    pivot_row = None
                    for r in range(rank, rows):
                        if mat_f[r, col] != 0:
                            pivot_row = r
                            break
                    if pivot_row is None:
                        continue
                    # Swap rows
                    if pivot_row != rank:
                        mat_f[rank, pivot_row] = mat_f[pivot_row, rank].copy()
                        mat_f[pivot_row, rank] = mat_f[rank, pivot_row].copy()
                    # Eliminate
                    pivot = mat_f[rank, col]
                    for r in range(rank + 1, rows):
                        if mat_f[r, col] != 0:
                            factor = mat_f[r, col]
                            for c in range(col, cols):
                                mat_f[r, c] = mat_f[r, c] * pivot - factor * mat_f[rank, c]
                    rank += 1
                return rank

            exact_rk = exact_rank(M_int)
            num_deps = rank - exact_rk

            if num_deps > 0:
                # Find null vectors using exact arithmetic
                # We'll use a simpler approach: try to eliminate columns one by one
                # using the exact rank information

                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()

                for iteration in range(num_deps + 2):
                    if time.time() > deadline - 25:
                        break

                    remaining = [c for c in range(rank) if c not in eliminated]
                    if len(remaining) <= 1:
                        break

                    M_sub = M[:, remaining]

                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break

                    tol_sub = 1e-9
                    eff_rank_sub = int(np.sum(s_sub > tol_sub))

                    if eff_rank_sub >= len(remaining):
                        break

                    null_sub = Vt_sub[eff_rank_sub:, :]

                    # Pick the null vector with the largest max coefficient
                    best_col = None
                    best_score = -1
                    best_null_idx = 0

                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            score = abs(nv[local_c])
                            if score > best_score and score > 1e-8:
                                best_score = score
                                best_col = global_c
                                best_null_idx = nv_idx

                    if best_col is None:
                        break

                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]

                    if abs(pivot_val) < 1e-10:
                        continue

                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]

                    eliminated.add(best_col)

                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]

                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass

    # Strategy 3: Try to find a rank-40 decomposition by exploiting block structure
    # The 6 Strassen blocks share structure - look for cross-block dependencies
    try:
        if time.time() < deadline - 20:
            # Each block has 7 terms. Try to find linear combinations across blocks
            # that sum to zero in the tensor space.

            # For each pair of blocks, try to find a dependency
            block_starts = [0, 7, 14, 21, 28, 35]
            block_size = 7

            # Try to find 2-term dependencies between blocks
            found_deps = []
            for i in range(6):
                for j in range(i + 1, 6):
                    if time.time() > deadline - 15:
                        break

                    # Get the 7x7 submatrix for blocks i and j
                    cols_i = block_starts[i] + list(range(block_size))
                    cols_j = block_starts[j] + list(range(block_size))
                    all_cols = cols_i + cols_j

                    M_pair = M[:, all_cols]

                    try:
                        _, s_pair, Vt_pair = np.linalg.svd(M_pair, full_matrices=False)
                    except np.linalg.LinAlgError:
                        continue

                    eff_rank_pair = int(np.sum(s_pair > 1e-9))

                    if eff_rank_pair < len(all_cols):
                        # There's a dependency in this pair
                        null_pair = Vt_pair[eff_rank_pair:, :]
                        for nv_idx in range(null_pair.shape[0]):
                            nv = null_pair[nv_idx]
                            # Check if this gives a useful elimination
                            if np.max(np.abs(nv)) > 1e-6:
                                found_deps.append((i, j, nv))

            # Try to use found dependencies to eliminate terms
            if found_deps:
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()

                for i, j, nv in found_deps:
                    if time.time() > deadline - 15:
                        break

                    cols_i = block_starts[i] + list(range(block_size))
                    cols_j = block_starts[j] + list(range(block_size))
                    all_cols = cols_i + cols_j

                    # Check which columns are still active
                    active_cols = [c for c in all_cols if c not in eliminated]
                    if len(active_cols) < 2:
                        continue

                    # Map nv to active columns
                    nv_active = []
                    for c in active_cols:
                        idx_in_all = all_cols.index(c)
                        nv_active.append(nv[idx_in_all])
                    nv_active = np.array(nv_active)

                    if np.max(np.abs(nv_active)) < 1e-8:
                        continue

                    # Find pivot
                    pivot_idx = np.argmax(np.abs(nv_active))
                    pivot_val = nv_active[pivot_idx]

                    if abs(pivot_val) < 1e-10:
                        continue

                    pivot_col = active_cols[pivot_idx]

                    # Eliminate
                    for idx, col in enumerate(active_cols):
                        if col == pivot_col:
                            continue
                        coeff = -nv_active[idx] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, col] += coeff * new_U[:, pivot_col]
                            new_V[:, col] += coeff * new_V[:, pivot_col]
                            new_W[:, col] += coeff * new_W[:, pivot_col]

                    eliminated.add(pivot_col)

                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]

                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass

    # Strategy 4: Aggressive multi-pass elimination with different tolerances
    try:
        if time.time() < deadline - 15:
            for tol_try in [1e-8, 1e-7, 1e-6, 1e-5]:
                if time.time() > deadline - 10:
                    break

                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()

                for iteration in range(10):
                    if time.time() > deadline - 10:
                        break

                    remaining = [c for c in range(rank) if c not in eliminated]
                    if len(remaining) <= 1:
                        break

                    M_sub = M[:, remaining]

                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break

                    eff_rank_sub = int(np.sum(s_sub > tol_try))

                    if eff_rank_sub >= len(remaining):
                        break

                    null_sub = Vt_sub[eff_rank_sub:, :]

                    best_col = None
                    best_score = -1
                    best_null_idx = 0

                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            score = abs(nv[local_c])
                            if score > best_score and score > 1e-8:
                                best_score = score
                                best_col = global_c
                                best_null_idx = nv_idx

                    if best_col is None:
                        break

                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]

                    if abs(pivot_val) < 1e-10:
                        continue

                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]

                    eliminated.add(best_col)

                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]

                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
