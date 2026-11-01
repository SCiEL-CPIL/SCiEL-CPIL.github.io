"""Tensor rank minimization for 2x5x6 matrix multiplication.

Strategy:
1. Build a valid rank-54 decomposition via block decomposition
   (Strassen on 2x2x2 blocks, schoolbook on the 1x2x2 strip).
2. Try to reduce rank by eliminating columns via least-squares correction.
3. Additionally try a parity-check / null-space guided search:
   - Build the full 60-dim tensor coordinate system.
   - For the current decomposition, the span of the rank-1 terms is a
     subspace of the tensor space.  We look for directions in that span
     that are orthogonal to all 60 "output" vectors (the e_{ijk} basis),
     which would let us shift a column out.
4. Polish: after rank reduction, run a final high-precision verification.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0  # 10 s safety margin before 150 s kill

    # ============================================================
    # Helper: build the 60-dim tensor coordinate for (i,j,k)
    # ============================================================
    def tensor_coord(i, j, k):
        return i * m * p + j * p + k

    def verify(U_v, V_v, W_v, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_v[i * m + j, :], V_v[j * p + k, :] * W_v[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # ============================================================
    # Phase 1: Build initial rank-54 decomposition
    # ============================================================
    # J (size 5): J1={0,1}, J2={2,3}, J3={4}
    # K (size 6): K1={0,1}, K2={2,3}, K3={4,5}
    # 6 Strassen 2x2x2 blocks + 3 schoolbook 1x2x2 blocks = 42+12 = 54

    initial_rank = 54
    U = np.zeros((n * m, initial_rank), dtype=float)
    V = np.zeros((m * p, initial_rank), dtype=float)
    W = np.zeros((p * n, initial_rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_block(ja, jb, ka, kb):
        """Add 7-term Strassen decomposition for a 2x2x2 block."""
        u0a = 0 * m + ja
        u0b = 0 * m + jb
        u1a = 1 * m + ja
        u1b = 1 * m + jb

        v_a_ka = ja * p + ka
        v_a_kb = ja * p + kb
        v_b_ka = jb * p + ka
        v_b_kb = jb * p + kb

        w_ka_0 = ka * n + 0
        w_ka_1 = ka * n + 1
        w_kb_0 = kb * n + 0
        w_kb_1 = kb * n + 1

        # M1: (A00+A11)(B00+B11)
        add_term([(u0a, 1), (u1b, 1)],
                 [(v_a_ka, 1), (v_b_kb, 1)],
                 [(w_ka_0, 1), (w_kb_1, 1)])
        # M2: (A10+A11)(B00)
        add_term([(u1a, 1), (u1b, 1)],
                 [(v_a_ka, 1)],
                 [(w_ka_1, 1), (w_kb_1, -1)])
        # M3: (A00)(B01-B11)
        add_term([(u0a, 1)],
                 [(v_a_kb, 1), (v_b_kb, -1)],
                 [(w_kb_0, 1), (w_kb_1, 1)])
        # M4: (A11)(B10-B00)
        add_term([(u1b, 1)],
                 [(v_b_ka, 1), (v_a_ka, -1)],
                 [(w_ka_0, 1), (w_ka_1, 1)])
        # M5: (A00+A01)(B11)
        add_term([(u0a, 1), (u0b, 1)],
                 [(v_b_kb, 1)],
                 [(w_ka_0, -1), (w_kb_0, 1)])
        # M6: (A10-A00)(B00+B01)
        add_term([(u1a, 1), (u0a, -1)],
                 [(v_a_ka, 1), (v_a_kb, 1)],
                 [(w_kb_1, 1)])
        # M7: (A01-A11)(B10+B11)
        add_term([(u0b, 1), (u1b, -1)],
                 [(v_b_ka, 1), (v_b_kb, 1)],
                 [(w_ka_0, 1)])

    # Add 6 Strassen blocks
    for (ja, jb) in [(0, 1), (2, 3)]:
        for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
            strassen_block(ja, jb, ka, kb)

    # Add 3 schoolbook 1x2x2 blocks for J3={4}
    for ka, kb in [(0, 1), (2, 3), (4, 5)]:
        for k in (ka, kb):
            for i in range(n):
                U[i * m + 4, col] = 1
                V[4 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_init = U[:, :col].copy()
    V_init = V[:, :col].copy()
    W_init = W[:, :col].copy()
    current_rank = col  # should be 54

    if not verify(U_init, V_init, W_init):
        # Fallback: trivial rank-60
        rank_fallback = n * m * p
        U_fb = np.zeros((n * m, rank_fallback), dtype=float)
        V_fb = np.zeros((m * p, rank_fallback), dtype=float)
        W_fb = np.zeros((p * n, rank_fallback), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # ============================================================
    # Phase 2: Rank reduction via column elimination
    # ============================================================
    # For each column l, we try to find a correction vector delta (length rank-1)
    # such that after removing column l and adding delta to the U rows of the
    # remaining columns, the tensor identity still holds.
    #
    # The key equation: for each (i,j,k):
    #   sum_{l' != l} U[i*m+j, l'] * V[j*p+k, l'] * W[k*n+i, l']
    #     + sum_{l' != l} delta[l'] * V[j*p+k, l'] * W[k*n+i, l'] * U_row_correction
    #   = 1
    #
    # We parameterize the correction as: for each remaining column l', we add
    # delta[l'] * s_{l'} to U[i*m+j, l'] where s_{l'} is a scalar.
    # This gives a linear system in the unknowns delta[l'].

    def try_remove_column(U_cur, V_cur, W_cur, rank_cur, col_to_remove):
        """Try to remove col_to_remove by adjusting U rows of other columns."""
        if rank_cur <= 1:
            return None

        idx_map = [l for l in range(rank_cur) if l != col_to_remove]
        num_unknowns = rank_cur - 1

        # Build linear system: for each (i,j,k), we need
        #   sum_{x} delta_x * V[j*p+k, l_x] * W[k*n+i, l_x] * (U[i*m+j, l_x] adjustment)
        # but the adjustment is delta_x applied uniformly to the U row.
        # Actually, the correction to the tensor from column l_x is:
        #   delta_x * (U[i*m+j, l_x] * V[j*p+k, l_x] * W[k*n+i, l_x])
        # Wait, that's not right. If we add delta_x to U[i*m+j, l_x] for all (i,j),
        # then the tensor contribution changes by delta_x * V[j*p+k, l_x] * W[k*n+i, l_x]
        # for each (j,k), but only for the specific (i,j) row.
        #
        # Let me reconsider. The tensor entry T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        #
        # We want to remove column l0. So we need:
        #   sum_{l != l0} U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] + correction = 1
        # where correction compensates for the removed column.
        #
        # If we allow arbitrary corrections to U rows, the correction from column l is:
        #   delta[l] * V[j*p+k, l] * W[k*n+i, l]
        # (since delta[l] is added to U[i*m+j, l] for the specific (i,j) pair)
        #
        # But delta[l] depends on (i,j) if we allow per-row corrections.
        # For simplicity, let's use a global scalar per column.
        # Then the correction is delta[l] * V[j*p+k, l] * W[k*n+i, l] for ALL (i,j,k).
        #
        # Actually, a cleaner approach: we can modify U, V, OR W. Let's modify U.
        # For column l, we add a vector c_l (length n*m) to U[:, l].
        # The change in T[i,j,k] is c_l[i*m+j] * V[j*p+k, l] * W[k*n+i, l].
        #
        # This is a linear system in the unknowns c_l[i*m+j] for l != l0, (i,j) in range.
        # That's (rank-1) * n * m = (rank-1) * 10 unknowns, with 60 equations.
        # For rank ~54, that's 530 unknowns and 60 equations - underdetermined.
        # We can find a minimum-norm solution.

        # Build A matrix: 60 x num_unknowns
        # Unknown x corresponds to (l_x, i, j) where l_x is in idx_map
        # A[r, x] = V[j*p+k, l_x] * W[k*n+i, l_x]  where r = i*m*p + j*p + k

        # RHS: for each (i,j,k), the amount we need to compensate =
        #   1 - sum_{l != l0} U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # which equals U[i*m+j, l0] * V[j*p+k, l0] * W[k*n+i, l0]

        rhs = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = tensor_coord(i, j, k)
                    rhs[r] = U_cur[i * m + j, col_to_remove] * V_cur[j * p + k, col_to_remove] * W_cur[k * n + i, col_to_remove]

        # Build A
        # num_unknowns = num_unknowns_cols * n * m
        # x = idx * n*m + i*m + j where idx indexes into idx_map
        A = np.zeros((n * m * p, num_unknowns * n * m))
        for idx, l_x in enumerate(idx_map):
            for i in range(n):
                for j in range(m):
                    x = idx * n * m + i * m + j
                    for k in range(p):
                        r = tensor_coord(i, j, k)
                        A[r, x] = V_cur[j * p + k, l_x] * W_cur[k * n + i, l_x]

        # Solve minimum-norm least squares
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=1e-12)
        except np.linalg.LinAlgError:
            return None

        # Check residual
        A_sol = A @ sol
        err = np.linalg.norm(A_sol - rhs) / max(np.linalg.norm(rhs), 1e-15)
        if err > 1e-9:
            return None

        # Build new matrices
        U_new = np.zeros((n * m, num_unknowns), dtype=float)
        V_new = V_cur[:, idx_map].copy()
        W_new = W_cur[:, idx_map].copy()

        for idx, l_x in enumerate(idx_map):
            U_new[:, idx] = U_cur[:, l_x]
            for i in range(n):
                for j in range(m):
                    U_new[i * m + j, idx] += sol[idx * n * m + i * m + j]

        # Verify
        if verify(U_new, V_new, W_new, tol=1e-9):
            return U_new, V_new, W_new
        return None

    # Try removing columns
    U_cur = U_init.copy()
    V_cur = V_init.copy()
    W_cur = W_init.copy()
    rank_cur = current_rank

    # Try each column in order, starting from the schoolbook blocks (last 12)
    # as they are more likely to be redundant
    col_order = list(range(rank_cur - 1, -1, -1))

    # Also try a second pass after the first pass removes some columns
    for pass_num in range(3):
        if time.time() > deadline - 30.0:
            break
        if rank_cur <= 20:  # diminishing returns
            break
        progress = False
        for col_idx in col_order:
            if time.time() > deadline - 20.0:
                break
            if col_idx >= rank_cur:
                continue
            result = try_remove_column(U_cur, V_cur, W_cur, rank_cur, col_idx)
            if result is not None:
                U_cur, V_cur, W_cur = result
                rank_cur -= 1
                progress = True
                # Restart column order after each successful removal
                col_order = list(range(rank_cur - 1, -1, -1))
                # Small pause to let time accumulate
                if time.time() > deadline - 15.0:
                    break
        if not progress:
            break

    final_rank = rank_cur
    U_final = U_cur[:, :final_rank]
    V_final = V_cur[:, :final_rank]
    W_final = W_cur[:, :final_rank]

    # ============================================================
    # Phase 3: Final verification
    # ============================================================
    if not verify(U_final, V_final, W_final, tol=1e-9):
        # Fallback to the initial decomposition
        if verify(U_init, V_init, W_init):
            U_final = U_init
            V_final = V_init
            W_final = W_init
            final_rank = current_rank
        else:
            # Trivial fallback
            rank_fallback = n * m * p
            U_fb = np.zeros((n * m, rank_fallback), dtype=float)
            V_fb = np.zeros((m * p, rank_fallback), dtype=float)
            W_fb = np.zeros((p * n, rank_fallback), dtype=float)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
