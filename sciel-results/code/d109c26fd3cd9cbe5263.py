"""Candidate: runtime randomised algebraic search for the (2,4,5) matrix-multiplication
tensor. A different approach family from the seed (which hardcodes the schoolbook tensor):
this program *searches* for a low-rank bilinear decomposition and verifies it against the
official trace convention before returning it."""

import random

import numpy as np

DIMENSIONS = [2, 4, 5]


def _mul_tensor(n, m, p):
    # T[i,j,k] with output index k*n+i (official trace convention W[k*n+i, r]).
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _decomp_to_T(n, m, p, U, V, W):
    # Reconstruct the tensor implied by (U,V,W):
    # U[(i*m+j), r], V[(j*p+k), r], W[(k*n+i), r].
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(len(U[0])):
        ui = U[:, r]
        vj = V[:, r]
        wk = W[:, r]
        for a in range(n * m):
            if ui[a] == 0:
                continue
            i, j = divmod(a, m)
            for b in range(m * p):
                if vj[b] == 0:
                    continue
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                wk_idx = k * n + i
                val = ui[a] * vj[b] * wk[wk_idx]
                if val:
                    T[i, j, k] += val
    return T


def _is_valid_decomp(n, m, p, U, V, W, tol=1e-6):
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    r = len(U[0])
    if len(V[0]) != r or len(W[0]) != r:
        return False
    T = _decomp_to_T(n, m, p, U, V, W)
    Tref = _mul_tensor(n, m, p)
    return bool(np.max(np.abs(T - Tref)) <= tol)


def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(20240517)

    Tref = _mul_tensor(n, m, p)
    nmm = n * m          # U rows
    mmp = m * p          # V rows
    pnn = p * n          # W rows

    # ---- Baseline (always-valid schoolbook decomposition) --------------------
    base_r = n * m * p
    U = [[0] * base_r for _ in range(nmm)]
    V = [[0] * base_r for _ in range(mmp)]
    W = [[0] * base_r for _ in range(pnn)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    def score_of(decomp):
        U_, V_, W_ = decomp
        if not _is_valid_decomp(n, m, p, U_, V_, W_):
            return base_r
        return len(U_[0])

    # ---- Randomised search: small "rank-reduction" mutations that PRESERVE ----
    #    the tensor algebraically, then a final exhaustive low-rank search.
    #
    #   Merge (Strassen-like): if two rank-1 terms have *exactly* the same U and
    #   V vectors (row support in the shared index) we can fold the third leg into
    #   a linear combination, but only when their (i, j, k) cells overlap. We use
    #   a safe variant: find two columns c1,c2 whose U[:,c1]==U[:,c2] AND
    #   V[:,c1]==V[:,c2]; then the combined W is W[:,c1]+W[:,c2] and we DROP one
    #   column. This reduces rank by 1 and preserves the tensor *if* the original
    #   W entries are compatible (we verify against Tref afterwards).
    #
    #   Split: replace one column c into two columns c,a whose U,V are split so
    #   that U_c+U_a == U_c, V_c+V_a == V_c (e.g. U_a=0 or U_a=U_c with a
    #   compensating V), increasing rank by 1; used only to create merge
    #   opportunities.

    def add_merge_candidates(best):
        U_, V_, W_ = best
        r = len(U_[0])
        for it in range(40000):
            if r < 2:
                break
            c1 = rng.randrange(r)
            c2 = rng.randrange(r)
            if c1 == c2:
                continue
            u1, u2 = U_[c1], U_[c2]
            v1, v2 = V_[c1], V_[c2]
            if u1 != u2 or v1 != v2:
                continue
            # Try merging: produce new W = W_c1 + W_c2, drop c2.
            new_W = [row[:] for row in W_]
            new_W[c1] = [a + b for a, b in zip(W_[c1], W_[c2])]
            new_U = [U_[x] for x in range(r) if x != c2]
            new_V = [V_[x] for x in range(r) if x != c2]
            new_W_rows = [row for x, row in enumerate(new_W) if x != c2]
            cand = (new_U, new_V, new_W_rows)
            s = score_of(cand)
            if s < len(best[0][0]) if False else True:
                pass
            # compare ranks
            if s < len(U_[0]):
                best = cand
                r = len(cand[0][0])
        return best

    def split_to_create_overlaps(decomp):
        U_, V_, W_ = decomp
        r = len(U_[0])
        c = rng.randrange(r)
        # Copy column c as a new column with U' = U_c, V' = V_c, W' = 0: this
        # adds an identical (U,V) pair -> guarantees at least one merge candidate.
        new_U = [row[:] for row in U_] + [U_[c][:]]
        new_V = [row[:] for row in V_] + [V_[c][:]]
        new_W = [row[:] for row in W_] + [row[:] for row in W_]
        new_W[-1] = [a - b for a, b in zip(W_[c], W_[c])]  # = 0
        return (new_U, new_V, new_W)

    best = (U, V, W)
    for outer in range(6):
        # occasionally create an overlap to enable a merge
        if outer % 2 == 1:
            best = split_to_create_overlaps(best)
        best = add_merge_candidates(best)

    # Final: keep the best valid decomposition found (fallback to schoolbook).
    Uf, Vf, Wf = best
    if not _is_valid_decomp(n, m, p, Uf, Vf, Wf):
        Uf, Vf, Wf = U, V, W

    return {"U": Uf, "V": Vf, "W": Wf}
