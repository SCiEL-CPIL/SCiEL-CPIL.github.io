import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 118  # safety margin from 150s wall-clock kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    # ------------------------------------------------------------------
    # Exact, fast validation using vectorised einsum.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] must equal 1 everywhere.
    # ------------------------------------------------------------------
    def validate(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
            return False
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return bool(np.max(np.abs(T - 1.0)) < tol)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # ------------------------------------------------------------------
    # Strassen 2x2x2 coefficient tables (local 2x2 indices 0..3).
    # ------------------------------------------------------------------
    U_coeffs = [
        {0: 1, 3: 1},   # M1: a, d
        {2: 1, 3: 1},   # M2: c, d
        {0: 1},         # M3: a
        {3: 1},         # M4: d
        {0: 1, 1: 1},   # M5: a, b
        {2: 1, 0: -1},  # M6: c, -a
        {1: 1, 3: -1},  # M7: b, -d
    ]
    V_coeffs = [
        {0: 1, 3: 1},   # M1: e, h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f, -h
        {2: 1, 0: -1},  # M4: g, -e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e, f
        {2: 1, 3: 1},   # M7: g, h
    ]
    W_coeffs = [
        {0: 1, 3: 1},   # M1: C00, C11
        {2: 1, 3: -1},  # M2: C10, -C11
        {1: 1, 3: 1},   # M3: C01, C11
        {0: 1, 2: 1},   # M4: C00, C10
        {0: -1, 1: 1},  # M5: -C00, C01
        {3: 1},         # M6: C11
        {0: 1},         # M7: C00
    ]

    # ------------------------------------------------------------------
    # Build a decomposition given a list of non-overlapping 2x2x2 Strassen
    # blocks (i0,j0,k0). Rank = 7*#blocks + schoolbook remainder.
    # ------------------------------------------------------------------
    def build_from_blocks(blocks):
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))
        n_school = n * m * p - len(covered)
        rank = 7 * len(blocks) + n_school
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for (i0, j0, k0) in blocks:
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_global = i0 + li // 2
                    j_global = j0 + li % 2
                    U[i_global * m + j_global, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_global = j0 + lj // 2
                    k_global = k0 + lj % 2
                    V[j_global * p + k_global, col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_global = i0 + lc // 2
                    k_global = k0 + lc % 2
                    W[k_global * n + i_global, col] += coeff
                col += 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
        assert col == rank, (col, rank)
        return {"U": U, "V": V, "W": W}, rank

    # ------------------------------------------------------------------
    # Enumerate non-overlapping Strassen block placements. A block at
    # (i0,j0,k0) requires i0 in {0,1}, j0 in {0,1}, k0 in {0,2,4,6}.
    # Blocks must be pairwise disjoint in the (i,j,k) grid.
    # ------------------------------------------------------------------
    def all_blocks():
        out = []
        for i0 in range(n - 1):      # 0,1
            for j0 in range(m - 1):  # 0,1
                for k0 in range(0, p - 1, 2):  # 0,2,4,6
                    out.append((i0, j0, k0))
        return out

    def blocks_disjoint(a, b):
        ai = {a[0], a[0] + 1}
        aj = {a[1], a[1] + 1}
        ak = {a[2], a[2] + 1}
        bi = {b[0], b[0] + 1}
        bj = {b[1], b[1] + 1}
        bk = {b[2], b[2] + 1}
        if ai & bi:
            return False
        if aj & bj:
            return False
        if ak & bk:
            return False
        return True

    def search_best_blocks():
        """Try all placements of 1..4 non-overlapping blocks, keep min rank."""
        cands = all_blocks()
        best_blocks = []
        best_rank = n * m * p
        # single blocks
        for b in cands:
            if 7 + (n * m * p - 8) < best_rank:
                best_rank = 7 + (n * m * p - 8)
                best_blocks = [b]
        # pairs
        for x in range(len(cands)):
            if time_left() < 20.0:
                break
            for y in range(x + 1, len(cands)):
                if not blocks_disjoint(cands[x], cands[y]):
                    continue
                r = 14 + (n * m * p - 16)
                if r < best_rank:
                    best_rank = r
                    best_blocks = [cands[x], cands[y]]
        # triples
        for x in range(len(cands)):
            if time_left() < 15.0:
                break
            for y in range(x + 1, len(cands)):
                if not blocks_disjoint(cands[x], cands[y]):
                    continue
                for z in range(y + 1, len(cands)):
                    if not blocks_disjoint(cands[x], cands[z]) or \
                       not blocks_disjoint(cands[y], cands[z]):
                        continue
                    r = 21 + (n * m * p - 24)
                    if r < best_rank:
                        best_rank = r
                        best_blocks = [cands[x], cands[y], cands[z]]
        # quadruples
        for x in range(len(cands)):
            if time_left() < 10.0:
                break
            for y in range(x + 1, len(cands)):
                if not blocks_disjoint(cands[x], cands[y]):
                    continue
                for z in range(y + 1, len(cands)):
                    if not blocks_disjoint(cands[x], cands[z]) or \
                       not blocks_disjoint(cands[y], cands[z]):
                        continue
                    for w in range(z + 1, len(cands)):
                        if not blocks_disjoint(cands[x], cands[w]) or \
                           not blocks_disjoint(cands[y], cands[w]) or \
                           not blocks_disjoint(cands[z], cands[w]):
                            continue
                        r = 28 + (n * m * p - 32)
                        if r < best_rank:
                            best_rank = r
                            best_blocks = [cands[x], cands[y], cands[z], cands[w]]
        return best_blocks, best_rank

    # ------------------------------------------------------------------
    # Greedy rank-1 peeling: peel a single monomial term if the remaining
    # tensor is a valid multiplication tensor. Safety net only.
    # ------------------------------------------------------------------
    def peel_greedy(factors, max_steps=200):
        U = np.array(factors["U"], dtype=float)
        V = np.array(factors["V"], dtype=float)
        W = np.array(factors["W"], dtype=float)
        r = U.shape[1]
        best = (U, V, W, r)
        for _ in range(max_steps):
            if time_left() < 8.0:
                break
            U3 = U[:, :r].reshape(n, m, r)
            V3 = V[:, :r].reshape(m, p, r)
            W3 = W[:, :r].reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            found = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T[i, j, k]) < 0.5:
                            continue
                        Tc = T.copy()
                        Tc[i, j, k] -= T[i, j, k]
                        if np.max(np.abs(Tc)) < 1e-8:
                            U[:, r] = 0.0
                            V[:, r] = 0.0
                            W[:, r] = 0.0
                            U[i * m + j, r] = T[i, j, k]
                            V[j * p + k, r] = 1.0
                            W[k * n + i, r] = 1.0
                            r += 1
                            best = (U, V, W, r)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return best

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    # 1) Search for the best block placement (fast, no validation needed).
    best_blocks, _ = search_best_blocks()

    # 2) Build and validate the best placement.
    best_factors = None
    if best_blocks:
        factors, rank = build_from_blocks(best_blocks)
        if validate(factors["U"], factors["V"], factors["W"]):
            best_factors = (factors["U"], factors["V"], factors["W"], rank)

    # Fallback: the fixed top-left embedding (known-good rank 68).
    if best_factors is None and time_left() > 15.0:
        fixed = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
        factors, rank = build_from_blocks(fixed)
        if validate(factors["U"], factors["V"], factors["W"]):
            best_factors = (factors["U"], factors["V"], factors["W"], rank)

    # 3) Greedy peeling (safety net; keeps validity).
    if best_factors is not None and time_left() > 12.0:
        peeled = peel_greedy({"U": best_factors[0], "V": best_factors[1],
                              "W": best_factors[2]})
        if validate(peeled[0], peeled[1], peeled[2]) and peeled[3] < best_factors[3]:
            best_factors = peeled

    if best_factors is None:
        return schoolbook()[0]

    U, V, W, r = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
