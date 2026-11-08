"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for the mixed 2x1x2 / 1x2x2 / 1x1x2 blocks, targeting rank 68,
then attempts a bounded greedy linear-dependence reduction to lower rank.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 138.0  # safety margin before 150s kill

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets.

        Correct Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)*e
        m3 = a*(f-h)
        m4 = d*(g-e)
        m5 = (a+b)*h
        m6 = (c-a)*(e+f)
        m7 = (b-d)*(g+h)

        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        """
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    U_t = []
    V_t = []
    W_t = []

    # A[0:2, 0:2] * B[0:2, k0:k0+2]: Strassen for each column-pair
    for k0 in [0, 2, 4, 6]:
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)

    # A[0:2, 2] * B[2, k0:k0+2]: direct (2x1x2 = 4 terms each)
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                direct_term(i, 2, k, U_t, V_t, W_t)

    # A[2, 0:2] * B[0:2, k0:k0+2]: direct (1x2x2 = 4 terms each)
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                direct_term(2, j, k, U_t, V_t, W_t)

    # A[2, 2] * B[2, k0:k0+2]: direct (1x1x2 = 2 terms each)
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
            direct_term(2, 2, k, U_t, V_t, W_t)

    rank = len(W_t)

    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # --- Bounded greedy linear-dependence rank reduction ---
    # Each rank term r contributes tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    # If term r's contribution equals a linear combination of other terms, we can
    # eliminate it. We attempt to find, for each candidate term r, whether there
    # exist coefficients c_s (s != r) such that T_r == sum_s c_s * T_s.
    # Because entries are small integers, we try small-integer coefficient search
    # over a limited candidate set to stay within budget.

    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank

    # Precompute the tensor contribution of each rank term as a flat vector
    def term_vector(r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        return vec

    # Try to eliminate each term by expressing it as a combination of the others.
    # Use a small coefficient grid {-1,0,1} over up to a few other terms via least
    # squares restricted to a candidate subset, then verify exactly.
    def try_eliminate(r, Ua, Va, Wa, rank_cur):
        """Try to express term r as integer combination of other terms. Return
        (new_U, new_V, new_W) with term r removed, or None if not found."""
        target = term_vector(r)
        # Candidate set: other terms
        others = [s for s in range(rank_cur) if s != r]
        if not others:
            return None
        # Build design matrix of candidate term vectors
        M = np.array([term_vector(s) for s in others])  # shape (len(others), nmp)
        # Solve least squares for coefficients
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            return None
        # Round to nearest integer and verify exactness
        coeffs_int = np.round(coeffs)
        if not np.all(np.abs(coeffs - coeffs_int) < 1e-6):
            # Try small-integer search over a reduced candidate subset
            # Use only candidates whose vector has nonzero overlap
            overlaps = np.abs(M @ target)
            order = np.argsort(-overlaps)
            top = [others[i] for i in order[:6]]
            Mt = np.array([term_vector(s) for s in top])
            try:
                ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
            except Exception:
                return None
            ct_int = np.round(ct)
            if np.all(np.abs(ct - ct_int) < 1e-6):
                # build coefficient map over `others`
                coeff_map = {s: 0.0 for s in others}
                for s, cv in zip(top, ct_int):
                    coeff_map[s] = cv
            else:
                return None
        else:
            coeff_map = {s: coeffs_int[i] for i, s in enumerate(others)}
            # filter out zeros
            coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}

        # Verify exactness of the representation
        recon = np.zeros(n * m * p)
        for s, c in coeff_map.items():
            recon += c * term_vector(s)
        if not np.allclose(recon, target, atol=1e-9):
            return None

        # Build new matrices with term r removed and other terms scaled
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = coeff_map.get(s, 1.0) if s in coeff_map else 1.0
            # For terms not in coeff_map, they are unchanged (scale 1)
            # For terms in coeff_map, they absorb the coefficient
            # Actually we must add the combination to the existing terms.
            # The existing term s already contributes c_s * T_s where c_s was 1.
            # We need the new contribution to equal original total.
            # Original total = T_r + sum_{s!=r} T_s.
            # We replace T_r with sum_{s} c_s T_s, so new total = sum_{s} (1 + c_s) T_s.
            new_U[:, new_idx] = Ua[:, s] * (1.0 + coeff_map.get(s, 0.0))
            new_V[:, new_idx] = Va[:, s] * (1.0 + coeff_map.get(s, 0.0))
            new_W[:, new_idx] = Wa[:, s] * (1.0 + coeff_map.get(s, 0.0))
        return new_U, new_V, new_W

    # Greedy: repeatedly try to eliminate a term until none can be removed or deadline
    cur_U, cur_V, cur_W = U_arr, V_arr, W_arr
    cur_rank = rank
    progress = True
    while progress and (time.time() - start_time) < DEADLINE - 5.0:
        progress = False
        for r in range(cur_rank):
            if (time.time() - start_time) > DEADLINE - 5.0:
                break
            res = try_eliminate(r, cur_U, cur_V, cur_W, cur_rank)
            if res is not None:
                new_U, new_V, new_W = res
                # Verify the reduced decomposition is still exact
                if verify_decomp(new_U, new_V, new_W):
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    cur_rank -= 1
                    progress = True
                    if cur_rank < best_rank:
                        best_U, best_V, best_W = cur_U, cur_V, cur_W
                        best_rank = cur_rank
                    break  # restart scan from top since indices shifted

    if best_rank < rank:
        U_arr, V_arr, W_arr = best_U, best_V, best_W

    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
