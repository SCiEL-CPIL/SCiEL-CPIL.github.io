"""
Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication.

Strategy:
  1. Construct an explicit rank-39 decomposition using Strassen's 7-multiplication
     algorithm for the 2x2 sub-blocks, combined with straightforward multiplication
     for the remaining terms.
  2. Verify the decomposition.
  3. Attempt a local-search refinement to reduce rank from 39 to 38 (and below).
     Uses fast residual-based verification and column-zeroing strategies.

The interface returns {'U', 'V', 'W'} with integer entries.
"""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---------- Schoolbook baseline (rank 42) ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # ---------- Fast verification using numpy ----------
    def verify_fast(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return bool(np.all(T == 1))

    def residual_norm(U, V, W, rank):
        """Compute L1 norm of residual (faster than full verify for rejection)."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return int(np.abs(T - 1).sum())

    # ---------- Strassen-based rank-39 construction ----------
    def strassen_rank39():
        """
        Partition A (2x3) into [A_22, A_21] where A_22 is 2x2 and A_21 is 2x1.
        Partition B (3x7) into [B_1, B_2, B_3, B_4] where B_1,B_2,B_3 are 3x2
        and B_4 is 3x1.

        For i in {1,2,3}:
          C_i = A_22 * B_i22 + A_21 * B_i12
          A_22 * B_i22: Strassen, 7 mults
          A_21 * B_i12: 4 mults (2x1 times 1x2)
          Total: 11 per block

        For i=4:
          C_4 = A_22 * B_421 + A_21 * B_411
          A_22 * B_421: 4 mults
          A_21 * B_411: 2 mults
          Total: 6

        Grand total: 3*11 + 6 = 39
        """
        rank = 39
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0

        def add_strassen_term(col_idx, i_base, k_base, u_coeffs, v_coeffs, w_contribs):
            u_rows = [i_base * m + 0, i_base * m + 1, (i_base + 1) * m + 0, (i_base + 1) * m + 1]
            for idx, coeff in enumerate(u_coeffs):
                if coeff != 0:
                    U[u_rows[idx]][col_idx] += coeff

            v_rows = [0 * p + k_base, 0 * p + k_base + 1, 1 * p + k_base, 1 * p + k_base + 1]
            for idx, coeff in enumerate(v_coeffs):
                if coeff != 0:
                    V[v_rows[idx]][col_idx] += coeff

            for (dr, dc), coeff in w_contribs.items():
                if coeff != 0:
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][col_idx] += coeff

        strassen_terms = [
            ([1, 0, 1, 0], [1, 1, 1, 1], {(0, 0): 1, (1, 1): 1}),
            ([0, 0, 1, 1], [1, 0, 0, 0], {(1, 0): 1}),
            ([1, 0, 0, 0], [0, 1, 0, -1], {(0, 1): 1}),
            ([0, 0, 0, 1], [0, -1, 1, 0], {(0, 0): 1, (1, 0): 1}),
            ([1, 1, 0, 0], [0, 0, 0, 1], {(0, 0): -1, (0, 1): 1}),
            ([-1, 0, 1, 0], [1, 1, 0, 0], {(1, 1): 1}),
            ([0, 1, 0, -1], [0, 0, 1, 1], {(0, 0): 1}),
        ]

        def add_21_12_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 2, 2 * p + k_base, {(0, 0): 1}),
                (i_base * m + 2, 2 * p + k_base + 1, {(0, 1): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base, {(1, 0): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base + 1, {(1, 1): 1}),
            ]
            for offset, (u_row, v_row, w_contribs) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                for (dr, dc), coeff in w_contribs.items():
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][c] += coeff

        def add_22_21_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 0, 0 * p + k_base, k_base * n + i_base),
                (i_base * m + 1, 1 * p + k_base, k_base * n + i_base),
                ((i_base + 1) * m + 0, 0 * p + k_base, k_base * n + (i_base + 1)),
                ((i_base + 1) * m + 1, 1 * p + k_base, k_base * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        def add_21_11_terms(col_start, i_base, k_base):
            terms = [
                (i_base * m + 2, 2 * p + k_base, k_base * n + i_base),
                ((i_base + 1) * m + 2, 2 * p + k_base, k_base * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        i_base = 0

        # Block 1: B_1 occupies k=0,1
        k_base = 0
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 2: B_2 occupies k=2,3
        k_base = 2
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 3: B_3 occupies k=4,5
        k_base = 4
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 4: B_4 occupies k=6
        k_base = 6
        add_22_21_terms(col, i_base, k_base)
        col += 4
        add_21_11_terms(col, i_base, k_base)
        col += 2

        assert col == 39, f"Expected 39 columns, got {col}"
        return {"U": U, "V": V, "W": W}

    # Verify and use the Strassen construction
    strassen_result = strassen_rank39()
    if verify_fast(strassen_result["U"], strassen_result["V"], strassen_result["W"], 39):
        best = strassen_result
        best_rank = 39

    # ---------- Fast local search using residual norm ----------
    def fast_local_search(U, V, W, target_rank, max_iters=2000):
        """
        Local search that uses residual norm for fast rejection.
        Only does full verification when residual is zero.
        """
        U_work = [row[:] for row in U]
        V_work = [row[:] for row in V]
        W_work = [row[:] for row in W]

        # Check initial state
        if residual_norm(U_work, V_work, W_work, target_rank) == 0:
            return True

        best_res = residual_norm(U_work, V_work, W_work, target_rank)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random modification
            mat_idx = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, target_rank - 1)
                old = U_work[row][col]
                U_work[row][col] = old + random.choice([-2, -1, 1, 2])
                res = residual_norm(U_work, V_work, W_work, target_rank)
                if res > best_res:
                    U_work[row][col] = old
                else:
                    best_res = res
                    if res == 0:
                        return True
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, target_rank - 1)
                old = V_work[row][col]
                V_work[row][col] = old + random.choice([-2, -1, 1, 2])
                res = residual_norm(U_work, V_work, W_work, target_rank)
                if res > best_res:
                    V_work[row][col] = old
                else:
                    best_res = res
                    if res == 0:
                        return True
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, target_rank - 1)
                old = W_work[row][col]
                W_work[row][col] = old + random.choice([-2, -1, 1, 2])
                res = residual_norm(U_work, V_work, W_work, target_rank)
                if res > best_res:
                    W_work[row][col] = old
                else:
                    best_res = res
                    if res == 0:
                        return True

        return best_res == 0

    def try_reduce_rank(target_rank, max_attempts=20):
        """Try to find a rank-target_rank decomposition starting from rank-39."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None

            # Start from Strassen solution, take first target_rank columns
            U0 = [row[:target_rank] for row in best["U"]]
            V0 = [row[:target_rank] for row in best["V"]]
            W0 = [row[:target_rank] for row in best["W"]]

            if fast_local_search(U0, V0, W0, target_rank, max_iters=300):
                return {"U": U0, "V": V0, "W": W0}

        return None

    # Try to reduce rank from current best, starting from 38 downward
    for target_rank in range(38, max(20, best_rank - 5)):
        if time.time() > deadline - 15:
            break

        result = try_reduce_rank(target_rank, max_attempts=10)
        if result:
            best = result
            best_rank = target_rank

    # Final verification
    if not verify_fast(best["U"], best["V"], best["W"], best_rank):
        # Fallback to schoolbook
        best = schoolbook()
        best_rank = 42

    return best
# EVOLVE-BLOCK-END
