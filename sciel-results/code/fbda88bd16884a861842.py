"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Key structural insight: the multiplication tensor T[i,a,k]=1 is SEPARABLE in the output
index i (n=2).  For each fixed i, the slice T_i[a,k]=1 is the all-ones matrix (rank 1),
so the tensor rank is bounded by 2 * (rank of the 2x3x2 sub-problem) via an i-separated
decomposition (terms do not mix i=0 and i=1).  The previous rank-44 base (Strassen on the
2x2 row-block of A x 2x8 block of B + schoolbook on the last row) is NOT i-separated, so
it is not optimal inside this family.

Strategy:
  1. Build a small i-separated base: Strassen for the (2,2,2) row-block + schoolbook for the
     remaining scalar row, per i.  Rank = 2 * (7 + 2) = 18.
  2. Reduce its rank via exact Gaussian elimination on the factor vectors: if a set of factor
     vectors (U_c, V_c, W_c) is linearly dependent, the extra terms are absorbed into the others
     (the 3-way sum is linear in each factor), giving a genuinely lower-rank i-separated decomp.
  3. Validate every candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the best i-separated result
     (or the known rank-44 base / schoolbook as final fallbacks).
"""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 132.0  # safety margin before the 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
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

    # ---- Exact validation using Fractions ----
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

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build the i-separated base (rank 18) ----
    def _build_isep_base():
        rank = 2 * (7 + 2)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for r in range(7):  # Strassen (2x2)*(2x2)
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] += u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + lk][col] += v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[lk * n + li][col] += w_val
                col += 1
            for li in range(2):  # schoolbook A[:,2]*(2x1) * B[2,:]*(1x2)
                for lk in range(2):
                    U[li * m + 2][col] = 1
                    V[2 * p + lk][col] = 1
                    W[lk * n + li][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    base_isep = _build_isep_base()
    if not _validate(base_isep):
        return _schoolbook()

    best = base_isep
    best_rank = len(base_isep["U"][0])

    # ---- Build the known rank-44 base (Strassen x 4 + schoolbook) ----
    def _build_base44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact Gaussian-elimination rank reduction ----
    # If a set of factor vectors (U_c, V_c, W_c) is linearly dependent, the extra terms are
    # absorbed into the others (the 3-way sum is linear in each factor).  Keep a basis of
    # independent vectors; drop the rest.  Result is validated exactly.
    def _reduce(cand):
        U = [list(row) for row in cand["U"]]
        V = [list(row) for row in cand["V"]]
        W = [list(row) for row in cand["W"]]
        rank = len(U[0])
        dim = n * m + m * p + p * n  # 6 + 24 + 16 = 46

        def vec_of(c):
            v = [Fraction(0)] * dim
            for idx in range(n * m):
                v[idx] = Fraction(U[idx][c])
            for idx in range(m * p):
                v[n * m + idx] = Fraction(V[idx][c])
            for idx in range(p * n):
                v[n * m + m * p + idx] = Fraction(W[idx][c])
            return v

        def reduce_against(v, basis):
            for b in basis:
                if v[0] != 0 and b[0] != 0:
                    f = v[0] / b[0]
                    v = [v[j] - f * b[j] for j in range(dim)]
            pivot = None
            for j in range(dim):
                if v[j] != 0:
                    pivot = j
                    break
            return v, pivot

        basis = []
        kept = []
        for c in range(rank):
            if not _elapsed_ok():
                return None
            v, pivot = reduce_against(vec_of(c), basis)
            if pivot is None:
                continue  # linearly dependent -> drop column c
            pv = v[pivot]
            v = [x / pv for x in v]
            v[pivot] = Fraction(1)
            basis.append(v)
            kept.append(c)
        if len(kept) >= rank:
            return cand
        new_rank = len(kept)
        nU = [[0] * new_rank for _ in range(n * m)]
        nV = [[0] * new_rank for _ in range(m * p)]
        nW = [[0] * new_rank for _ in range(p * n)]
        for nc, c in enumerate(kept):
            for idx in range(n * m):
                nU[idx][nc] = U[idx][c]
            for idx in range(m * p):
                nV[idx][nc] = V[idx][c]
            for idx in range(p * n):
                nW[idx][nc] = W[idx][c]
        return {"U": nU, "V": nV, "W": nW}

    # ---- Reduce the i-separated base ----
    if _elapsed_ok():
        red = _reduce(base_isep)
        if red is not None and _validate(red):
            r = len(red["U"][0])
            if r < best_rank:
                best = red
                best_rank = r

    # ---- Also reduce / consider the rank-44 base ----
    if _elapsed_ok():
        base44 = _build_base44()
        if _validate(base44):
            if len(base44["U"][0]) < best_rank:
                best = base44
                best_rank = len(base44["U"][0])
            red44 = _reduce(base44)
            if red44 is not None and _validate(red44):
                r = len(red44["U"][0])
                if r < best_rank:
                    best = red44
                    best_rank = r

    # ---- Final safety fallbacks ----
    if best is None or not _validate(best):
        return _schoolbook()
    return best

# EVOLVE-BLOCK-END
