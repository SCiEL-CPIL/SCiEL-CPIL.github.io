"""Strassen-based decomposition for 2x3x8 matrix multiplication with rank-reduction polish."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 vectors (7 terms)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: 4 Strassen blocks for A[:, 0:2] * B[0:2, :]
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: schoolbook for A[:, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Rank-reduction pass: try to eliminate linearly dependent columns
    def _try_reduce(U_mat, V_mat, W_mat, max_rank):
        """Try to reduce rank by removing linearly dependent terms."""
        import numpy as np
        from numpy.linalg import matrix_rank

        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)

        nU, nV, nW = U_arr.shape[0], V_arr.shape[0], W_arr.shape[0]
        cur_rank = U_arr.shape[1]

        # Track which columns are "active"
        active = list(range(cur_rank))
        kept_U = [U_arr[:, c] for c in active]
        kept_V = [V_arr[:, c] for c in active]
        kept_W = [W_arr[:, c] for c in active]

        # Try greedy removal: check if each column can be expressed as linear combo of others
        for attempt in range(50):
            if not _elapsed_ok():
                break
            if len(active) <= max_rank:
                break

            removed = False
            for idx_in_active in range(len(active) - 1, -1, -1):
                if not _elapsed_ok():
                    break
                if len(active) <= max_rank:
                    break
                c = active[idx_in_active]

                # Build matrix of all other active columns
                others = [a for j, a in enumerate(active) if j != idx_in_active]
                if len(others) < 1:
                    continue

                U_others = U_arr[:, others]
                V_others = V_arr[:, others]
                W_others = W_arr[:, others]

                # Check if column c is in span of others for each matrix
                # For U: is U[:,c] in colspace of U[:,others]?
                rank_others_U = matrix_rank(U_others)
                rank_with_U = matrix_rank(np.column_stack([U_others, U_arr[:, c]]))
                if rank_with_U > rank_others_U:
                    continue  # not in span for U

                rank_others_V = matrix_rank(V_others)
                rank_with_V = matrix_rank(np.column_stack([V_others, V_arr[:, c]]))
                if rank_with_V > rank_others_V:
                    continue

                rank_others_W = matrix_rank(W_others)
                rank_with_W = matrix_rank(np.column_stack([W_others, W_arr[:, c]]))
                if rank_with_W > rank_others_W:
                    continue

                # Column c appears linearly dependent in all three matrices
                # But we need to check the tensor product constraint still holds
                # This is complex; skip for safety
                pass

            if not removed:
                break

        return None  # No reduction found with this simple approach

    # Try a different reduction: look for pairs of columns that can be merged
    # by checking if their tensor contributions cancel
    def _try_pair_merge(U_mat, V_mat, W_mat):
        """Check if any two terms can be merged (their outer products sum to zero in some coordinate)."""
        import numpy as np

        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        cur_rank = U_arr.shape[1]

        # For each pair (a,b), check if there exist scalars x,y such that
        # x*U_a + y*U_b has support in a strict subset, etc.
        # This is expensive; just check simple cases where one is scalar multiple of other
        for a in range(cur_rank):
            if not _elapsed_ok():
                break
            for b in range(a + 1, cur_rank):
                # Check if U[:,a] is proportional to U[:,b]
                ua = U_arr[:, a]
                ub = U_arr[:, b]
                va = V_arr[:, a]
                vb = V_arr[:, b]
                wa = W_arr[:, a]
                wb = W_arr[:, b]

                # Skip if any is zero
                if np.allclose(ua, 0) or np.allclose(ub, 0):
                    continue
                if np.allclose(va, 0) or np.allclose(vb, 0):
                    continue
                if np.allclose(wa, 0) or np.allclose(wb, 0):
                    continue

                # Check proportionality
                try:
                    ratio_u = None
                    for i in range(len(ua)):
                        if abs(ua[i]) > 1e-10:
                            ratio_u = ub[i] / ua[i]
                            break
                    if ratio_u is None:
                        continue
                    if not np.allclose(ub, ratio_u * ua, atol=1e-9):
                        continue

                    ratio_v = None
                    for i in range(len(va)):
                        if abs(va[i]) > 1e-10:
                            ratio_v = vb[i] / va[i]
                            break
                    if ratio_v is None:
                        continue
                    if not np.allclose(vb, ratio_v * va, atol=1e-9):
                        continue

                    ratio_w = None
                    for i in range(len(wa)):
                        if abs(wa[i]) > 1e-10:
                            ratio_w = wb[i] / wa[i]
                            break
                    if ratio_w is None:
                        continue
                    if not np.allclose(wb, ratio_w * wa, atol=1e-9):
                        continue

                    # All proportional! We can merge.
                    # x*term_a + y*term_b = 0 means x + y*ratio = 0 in each factor
                    # So x = 1, y = -1/ratio_u (if ratio_u == ratio_v == ratio_w)
                    if abs(ratio_u - ratio_v) > 1e-9 or abs(ratio_u - ratio_w) > 1e-9:
                        continue

                    # Merge: subtract term_b/ratio from term_a
                    # New term: (U_a - U_b/r, V_a - V_b/r, W_a - W_b/r)
                    r = ratio_u
                    U_mat[[i for i in range(len(U_mat))]] = U_mat  # no change needed
                    # Actually modify in place
                    for i in range(len(U_mat)):
                        U_mat[i][a] = U_mat[i][a] - U_mat[i][b] / r
                        U_mat[i][b] = 0
                    for i in range(len(V_mat)):
                        V_mat[i][a] = V_mat[i][a] - V_mat[i][b] / r
                        V_mat[i][b] = 0
                    for i in range(len(W_mat)):
                        W_mat[i][a] = W_mat[i][a] - W_mat[i][b] / r
                        W_mat[i][b] = 0

                    # Now column b is all zeros, we can drop it
                    # Rebuild without column b
                    new_U = [[U_mat[i][j] for j in range(len(U_mat[0])) if j != b] for i in range(len(U_mat))]
                    new_V = [[V_mat[i][j] for j in range(len(V_mat[0])) if j != b] for i in range(len(V_mat))]
                    new_W = [[W_mat[i][j] for j in range(len(W_mat[0])) if j != b] for i in range(len(W_mat))]
                    return new_U, new_V, new_W

                except Exception:
                    continue

        return None

    # Try the pair merge reduction
    reduced = _try_pair_merge(U, V, W)
    if reduced is not None:
        U, V, W = reduced
        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate

    # If no reduction worked, return the valid rank-44 candidate
    return candidate

# EVOLVE-BLOCK-END
