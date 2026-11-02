import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 105  # Safety margin before 120s limit

    # Strassen decomposition for 2x2x2
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]

    def _build_strassen_3x4x8():
        """Block-Strassen decomposition: rank 88. Split n=3 into 2+1."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l in range(4):
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += coeff
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += coeff
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    U[2 * m + (2 * j1 + j2)][col] = 1
                    V[(2 * j1 + j2) * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_strassen_alt_split():
        """Alternative split: n=3 into 1+2. Top schoolbook, bottom Strassen."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += coeff
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += coeff
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition (trace convention)."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    def _build_signature_matrix(U, V, W):
        """Build S matrix where S[(i,j,k), r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        S = np.zeros((n * m * p, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = (i * m + j) * p + k
                    S[t, :] = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]
        return S

    def _prune_by_rank(U, V, W, tol=1e-9):
        """
        Find minimal generating set of columns by incremental linear-independence test.
        Returns list of column indices to keep.
        """
        S = _build_signature_matrix(U, V, W)
        rank = S.shape[1]

        keep = []
        basis = []
        for r in range(rank):
            col = S[:, r]
            if not basis:
                keep.append(r)
                basis.append(col.copy())
                continue
            B = np.stack(basis, axis=1)
            try:
                c, _, _, _ = np.linalg.lstsq(B, col, rcond=None)
                residual = np.linalg.norm(col - B @ c)
            except np.linalg.LinAlgError:
                keep.append(r)
                basis.append(col.copy())
                continue
            if residual > tol * 10:
                keep.append(r)
                basis.append(col.copy())
        return keep

    def _prune_greedy(U, V, W, tol=1e-9):
        """
        Greedy removal: repeatedly find a column closest to being in span of others,
        remove it if residual is small enough.
        """
        S = _build_signature_matrix(U, V, W)
        rank = S.shape[1]
        keep = list(range(rank))
        changed = True
        while changed and len(keep) > 1:
            changed = False
            best_r = -1
            best_res = tol * 5
            for pos, r in enumerate(keep):
                others = [q for q in keep if q != r]
                if len(others) == 0:
                    continue
                S_others = S[:, others]
                try:
                    c, _, _, _ = np.linalg.lstsq(S_others, S[:, r], rcond=None)
                    residual = np.linalg.norm(S[:, r] - S_others @ c)
                except np.linalg.LinAlgError:
                    continue
                if residual < best_res:
                    best_res = residual
                    best_r = r
            if best_r >= 0 and best_res < tol * 5:
                keep = [q for q in keep if q != best_r]
                changed = True
            else:
                break
        return keep

    def _try_rank_reduction(U, V, W, target_rank, tol=1e-9):
        """
        Try to reduce to target_rank by finding a subset of columns of that size
        whose span contains all original column signatures.
        Returns (U_red, V_red, W_red) or None if not possible.
        """
        S = _build_signature_matrix(U, V, W)
        rank = S.shape[1]
        if target_rank >= rank:
            return None

        # Greedy basis extraction to get a minimal set first
        keep = _prune_by_rank(U, V, W, tol)
        if len(keep) > target_rank:
            # Try to further reduce via greedy removal
            keep = _prune_greedy(U, V, W, tol)

        if len(keep) <= target_rank:
            U_red = [[row[r] for r in keep] for row in U]
            V_red = [[row[r] for r in keep] for row in V]
            W_red = [[row[r] for r in keep] for row in W]
            return U_red, V_red, W_red
        return None

    # Build candidate decompositions
    best_result = None
    best_rank = float('inf')

    candidates = []

    # Candidate 1: original Strassen split (n: 2+1)
    c1 = _build_strassen_3x4x8()
    if _verify(c1['U'], c1['V'], c1['W']) and _verify_half_integer(c1['U'], c1['V'], c1['W']):
        candidates.append((88, c1))

    # Candidate 2: alternative split (n: 1+2)
    if time.time() < deadline:
        c2 = _build_strassen_alt_split()
        if _verify(c2['U'], c2['V'], c2['W']) and _verify_half_integer(c2['U'], c2['V'], c2['W']):
            candidates.append((88, c2))

    # Try to reduce rank via linear-dependence pruning on each candidate
    for cand_rank, cand in list(candidates):
        if time.time() > deadline - 10:
            break

        # Prune by rank (basis extraction)
        try:
            keep = _prune_by_rank(cand['U'], cand['V'], cand['W'])
            if len(keep) < cand_rank:
                U_red = [[row[r] for r in keep] for row in cand['U']]
                V_red = [[row[r] for r in keep] for row in cand['V']]
                W_red = [[row[r] for r in keep] for row in cand['W']]
                if _verify(U_red, V_red, W_red) and _verify_half_integer(U_red, V_red, W_red):
                    candidates.append((len(keep), {"U": U_red, "V": V_red, "W": W_red}))
        except Exception:
            pass

        # Also try greedy pruning
        try:
            keep_g = _prune_greedy(cand['U'], cand['V'], cand['W'])
            if len(keep_g) < cand_rank:
                U_red = [[row[r] for r in keep_g] for row in cand['U']]
                V_red = [[row[r] for r in keep_g] for row in cand['V']]
                W_red = [[row[r] for r in keep_g] for row in cand['W']]
                if _verify(U_red, V_red, W_red) and _verify_half_integer(U_red, V_red, W_red):
                    candidates.append((len(keep_g), {"U": U_red, "V": V_red, "W": W_red}))
        except Exception:
            pass

    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, best_result = candidates[0]
    else:
        # Fallback
        best_result = _build_strassen_3x4x8()
        best_rank = 88

    return best_result
# EVOLVE-BLOCK-END
