import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Precompute the valid tensor indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i * m + j, j * p + k, k * n + i))
    valid_indices = np.array(valid_indices)  # shape (48, 3)

    # ---- Build Strassen rank-42 decomposition ----
    def add_strassen_2x2x2(U, V, W, i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        # M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        # M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0
        # M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0
        # M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0
        # M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0
        # M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0
        # M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    RANK_INIT = 42
    U = np.zeros((nm, RANK_INIT), dtype=np.float64)
    V = np.zeros((mp, RANK_INIT), dtype=np.float64)
    W = np.zeros((pn, RANK_INIT), dtype=np.float64)

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_2x2x2(U, V, W, i_start, j_start, k_start, offset)
                offset += 7

    # ---- Vectorized fast verification ----
    def verify_fast(Um, Vm, Wm, tol=1e-9):
        """Check all 48 valid entries vectorized."""
        r = Um.shape[1]
        # For each valid (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
        # Extract rows
        U_rows = Um[valid_indices[:, 0]]  # (48, r)
        V_rows = Vm[valid_indices[:, 1]]  # (48, r)
        W_rows = Wm[valid_indices[:, 2]]  # (48, r)
        # Element-wise product then sum over r
        products = U_rows * V_rows * W_rows  # (48, r)
        sums = products.sum(axis=1)  # (48,)
        return np.all(np.abs(sums - 1.0) < tol)

    # ---- Exact verification using Fraction ----
    def verify_exact(Um, Vm, Wm):
        r = Um.shape[1]
        Uf = [[Fraction(int(round(Um[i, j]))) for j in range(r)] for i in range(nm)]
        Vf = [[Fraction(int(round(Vm[i, j]))) for j in range(r)] for i in range(mp)]
        Wf = [[Fraction(int(round(Wm[i, j]))) for j in range(r)] for i in range(pn)]
        valid_set = set((a, b, c) for a, b, c in valid_indices)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = Fraction(0)
                    for r_idx in range(r):
                        s += Uf[a][r_idx] * Vf[b][r_idx] * Wf[c][r_idx]
                    want = Fraction(1) if (a, b, c) in valid_set else Fraction(0)
                    if s != want:
                        return False
        return True

    # Sanity check base decomposition
    if not verify_fast(U, V, W):
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=np.float64)
        V = np.zeros((mp, rank0), dtype=np.float64)
        W = np.zeros((pn, rank0), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = best_U.shape[1]

    # ---- Column removal: try each column, vectorized check ----
    def try_remove_column(Um, Vm, Wm):
        r = Um.shape[1]
        for col in range(r):
            if time.time() > deadline - 5:
                return None
            U2 = np.delete(Um, col, axis=1)
            V2 = np.delete(Vm, col, axis=1)
            W2 = np.delete(Wm, col, axis=1)
            if verify_fast(U2, V2, W2):
                return (U2, V2, W2)
        return None

    # Greedy removal: keep removing until stuck
    while time.time() < deadline - 5:
        res = try_remove_column(best_U, best_V, best_W)
        if res is None:
            break
        best_U, best_V, best_W = res
        best_rank = best_U.shape[1]

    # ---- Second pass: try different column orderings ----
    if time.time() < deadline - 5:
        r = best_U.shape[1]
        # Try removing in reverse order
        for start_order in [range(r), range(r - 1, -1, -1)]:
            if time.time() > deadline - 5:
                break
            cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
            cur_rank = cur_U.shape[1]
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for col in start_order:
                    if col >= cur_U.shape[1]:
                        continue
                    if time.time() > deadline - 5:
                        break
                    U2 = np.delete(cur_U, col, axis=1)
                    V2 = np.delete(cur_V, col, axis=1)
                    W2 = np.delete(cur_W, col, axis=1)
                    if verify_fast(U2, V2, W2):
                        cur_U, cur_V, cur_W = U2, V2, W2
                        cur_rank = cur_U.shape[1]
                        changed = True
                        break
                # If we changed, restart the scan
            if cur_rank < best_rank:
                best_U, best_V, best_W = cur_U, cur_V, cur_W
                best_rank = cur_rank

    # ---- Try pairwise column merging: replace two columns with their sum/difference ----
    if time.time() < deadline - 10:
        r = best_U.shape[1]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            r = best_U.shape[1]
            for c1 in range(r):
                if time.time() > deadline - 10:
                    break
                for c2 in range(c1 + 1, r):
                    if time.time() > deadline - 10:
                        break
                    # Try: replace columns c1,c2 with (c1+c2) and (c1-c2)
                    # Actually, we want to check if one column is redundant given the other
                    # Simpler: just try removing c1 or c2 (already done above)
                    # Alternative: check if col c1 is in span of others (linear dependency)
                    pass
            # Linear dependency check: for each column, check if it's a linear combo of others
            r = best_U.shape[1]
            if r > 1:
                # Build a matrix where each row is a "signature" of the column
                # Column r contributes: U[:,r] * V[:,r] * W[:,r] to the tensor
                # Two columns are linearly dependent if their contribution tensors are proportional
                # This is complex; skip for now
                pass

    # ---- Final exact verification ----
    if time.time() < deadline - 2:
        if verify_exact(best_U, best_V, best_W):
            final_U, final_V, final_W = best_U, best_V, best_W
        else:
            # Fall back to the base rank-42
            final_U, final_V, final_W = U, V, W
    else:
        final_U, final_V, final_W = U, V, W

    # Ensure entries are clean integers (they should be ±1 or 0)
    final_U = np.round(final_U)
    final_V = np.round(final_V)
    final_W = np.round(final_W)

    return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
