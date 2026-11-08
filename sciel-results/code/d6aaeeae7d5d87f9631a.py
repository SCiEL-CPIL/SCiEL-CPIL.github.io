import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin vs 150s kill

    # ---------- helpers ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def target_set():
        t = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t.add((i * m + j, j * p + k, k * n + i))
        return t

    TGT = target_set()

    def verify_exact(U, V, W, rank):
        """Exact verification using Fraction; entries must be half-integers."""
        # check half-integer
        for M in (U, V, W):
            for row in M:
                for x in row:
                    fx = Fraction(x) if not isinstance(x, Fraction) else x
                    if (fx * 2).denominator != 1:
                        return False
        # build tensor
        T = {}
        for r in range(rank):
            for a in range(n * m):
                ua = U[a][r]
                if ua == 0:
                    continue
                ua_f = Fraction(ua) if not isinstance(ua, Fraction) else ua
                for b in range(m * p):
                    vb = V[b][r]
                    if vb == 0:
                        continue
                    vb_f = Fraction(vb) if not isinstance(vb, Fraction) else vb
                    for c in range(p * n):
                        wc = W[c][r]
                        if wc == 0:
                            continue
                        wc_f = Fraction(wc) if not isinstance(wc, Fraction) else wc
                        key = (a, b, c)
                        T[key] = T.get(key, Fraction(0)) + ua_f * vb_f * wc_f
        # check target entries
        for key in TGT:
            if T.get(key, Fraction(0)) != Fraction(1):
                return False
        # check non-target entries are zero
        for key, val in T.items():
            if key not in TGT and val != 0:
                return False
        return True

    # ---------- Strassen 2x2x2 core ----------
    # Each term: (U_coeffs, V_coeffs, W_coeffs) where each is list of (row, col, coeff)
    STRASSEN = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,-1),(1,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1)]),
    ]

    # 4 blocks: (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
    BLOCKS = [
        (0, 0, 0, 0, 0, 0),  # A11*B11 -> C11
        (0, 0, 0, 2, 2, 0),  # A11*B12 -> C12
        (0, 2, 2, 0, 0, 0),  # A12*B21 -> C11
        (0, 2, 2, 2, 2, 0),  # A12*B22 -> C12
    ]

    def build_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for (ar_off, ac_off, br_off, bc_off, cr_off, cc_off) in BLOCKS:
            for (u_c, v_c, w_c) in STRASSEN:
                for (ar, ac, coeff) in u_c:
                    U[ar_off + ar, ac_off + ac] = coeff
                for (br, bc, coeff) in v_c:
                    V[br_off + br, bc_off + bc] = coeff
                for (cr, cc, coeff) in w_c:
                    W[cr_off + cr, cc_off + cc] = coeff
                col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # ---------- Search for lower rank via structured perturbation ----------
    def try_rank_with_corrections(base_U, base_V, base_W, base_rank, target_rank, max_extra=4):
        """Try to reduce rank by zeroing some columns and adjusting others."""
        if time.time() > deadline - 10:
            return None
        # This is a heuristic: try removing columns from the Strassen decomposition
        # and see if the remaining tensor still matches (it won't, but check)
        # For now, just verify the base decomposition
        return None

    # ---------- Main search ----------
    best_result, best_rank = schoolbook()

    # Try Strassen rank 28
    if time.time() < deadline - 20:
        strassen, r28 = build_strassen_28()
        if verify_exact(strassen["U"], strassen["V"], strassen["W"], r28):
            best_result = strassen
            best_rank = r28
            print(f"Strassen rank {r28} verified exactly")
        else:
            print("Strassen rank 28 failed exact verification")

    # Try to find rank < 28 via exhaustive small-perturbation search
    # Strategy: take the Strassen decomposition and try to merge columns
    # (i.e., find linear dependencies among the rank-28 terms)
    if best_rank > 24 and time.time() < deadline - 15:
        strassen, r28 = build_strassen_28()
        U_arr = np.array(strassen["U"], dtype=float)
        V_arr = np.array(strassen["V"], dtype=float)
        W_arr = np.array(strassen["W"], dtype=float)

        # Check for linear dependencies in the U matrix columns
        # If two columns of U are linearly dependent, we might merge them
        # Use SVD to find near-dependencies
        try:
            # For each pair of columns, check if they're proportional
            for r1 in range(r28):
                if time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, r28):
                    # Check if column r1 and r2 of U are proportional
                    u1 = U_arr[:, r1]
                    u2 = U_arr[:, r2]
                    if np.allclose(u1, 0) or np.allclose(u2, 0):
                        continue
                    # Check proportionality
                    nonzero1 = u1 != 0
                    nonzero2 = u2 != 0
                    if not np.all(nonzero1 == nonzero2):
                        continue
                    if np.any(nonzero1):
                        ratio = u1[nonzero1] / u2[nonzero2]
                        if np.allclose(ratio, ratio[0], atol=1e-10):
                            # Proportional! Check V and W
                            v1 = V_arr[:, r1]
                            v2 = V_arr[:, r2]
                            w1 = W_arr[:, r1]
                            w2 = W_arr[:, r2]
                            # If V and W are also proportional, we can merge
                            if np.allclose(v1, ratio[0] * v2, atol=1e-10) and \
                               np.allclose(w1, (1/ratio[0]) * w2, atol=1e-10):
                                # Merge: subtract
                                new_rank = r28 - 1
                                # Create new matrices with merged column
                                U_new = np.delete(U_arr, r2, axis=1)
                                V_new = np.delete(V_arr, r2, axis=1)
                                W_new = np.delete(W_arr, r2, axis=1)
                                # Verify
                                U_l = U_new.tolist()
                                V_l = V_new.tolist()
                                W_l = W_new.tolist()
                                if verify_exact(U_l, V_l, W_l, new_rank):
                                    best_result = {"U": U_l, "V": V_l, "W": W_l}
                                    best_rank = new_rank
                                    print(f"Merged to rank {new_rank}")
                                    break
                if best_rank < r28:
                    break
        except Exception as e:
            print(f"Merge search error: {e}")

    # Try rank 24 via random half-integer search (small budget)
    if best_rank > 24 and time.time() < deadline - 5:
        import random
        random.seed(42)
        # Try a few random half-integer matrices of rank 24
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            rank = 24
            U = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
            if verify_exact(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break

    # Ensure return format matches interface
    result = {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
    return result
# EVOLVE-BLOCK-END
