"""Matrix multiplication 4x4x9: block-Winograd (Strassen on 2x2 row-blocks)."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

# ---------------------------------------------------------------------------
# Conventions:
#   U row index = i*m + j   (A is n x m)
#   V row index = j*p + k   (B is m x p)
#   W row index = k*n + i   (C is p x n, trace convention)
# Tensor equation: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
# ---------------------------------------------------------------------------

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # entries must be integer or half-integer
    for arr in (Ua, Va, Wa):
        d = arr * 2
        if not np.allclose(d, np.round(d), atol=1e-9):
            return False
    # exact tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.sum(Ua[i * m + j, :] * Va[j * p + k, :] * Wa[k * n + i, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


# Strassen 7-factor decomposition for a 2x2 (row) block, shared across `pcol` columns.
# Local A: 2 x (m*2)  -> global A row i0+di, col j0+dj
# Local B: (m*2) x pcol -> global B row j0+dj, col k0+dk
# Local C: pcol x 2    -> global C col k0+dk, row i0+di
# Strassen factors (di, dj) on A, (dj, dk) on B, (dk, di) on C:
_STRASSEN = [
    # (u_terms, v_terms, w_terms)
    # M1 = (A00+A11)(B00+B11) -> C00 +1, C11 +1
    ([((0, 0), 1), ((1, 1), 1)],
     [((0, 0), 1), ((1, 1), 1)],
     [((0, 0), 1), ((1, 1), 1)]),
    # M2 = (A10+A11)B00 -> C10 +1
    ([((1, 0), 1), ((1, 1), 1)],
     [((0, 0), 1)],
     [((0, 1), 1)]),
    # M3 = A00(B01-B11) -> C01 +1
    ([((0, 0), 1)],
     [((0, 1), 1), ((1, 1), -1)],
     [((1, 0), 1)]),
    # M4 = A11(B10-B00) -> C10 +1, C00 +1
    ([((1, 1), 1)],
     [((1, 0), 1), ((0, 0), -1)],
     [((0, 1), 1), ((0, 0), 1)]),
    # M5 = (A00+A01)B11 -> C01 +1, C00 -1
    ([((0, 0), 1), ((0, 1), 1)],
     [((1, 1), 1)],
     [((1, 0), 1), ((0, 0), -1)]),
    # M6 = (A10-A00)(B00+B01) -> C11 +1
    ([((1, 0), 1), ((0, 0), -1)],
     [((0, 0), 1), ((0, 1), 1)],
     [((1, 1), 1)]),
    # M7 = (A01-A11)(B10+B11) -> C00 +1
    ([((0, 1), 1), ((1, 1), -1)],
     [((1, 0), 1), ((1, 1), 1)],
     [((0, 0), 1)]),
]


def _block_winograd(n, m, p):
    """Partition into 2x2 blocks of (i,j) and apply rank-7 Strassen on the 2x2
    row-block dimension. Each (i0,j0) block is independent -> exact decomposition.

    Block (i0, j0): A rows i0..i0+1, cols j0..j0+1 (2x2)
                    B rows j0..j0+1, all p cols (2 x p)
                    C cols all p, rows i0..i0+1 (p x 2)
    Rank per block = 7 * p.
    """
    ni, nj = n // 2, m // 2
    rank = ni * nj * 7 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for bi in range(ni):
        for bj in range(nj):
            i0 = 2 * bi
            j0 = 2 * bj
            for (u_t, v_t, w_t) in _STRASSEN:
                # A: (di, dj) -> global row i0+di, col j0+dj  -> U index (i0+di)*m + (j0+dj)
                for (di, dj), val in u_t:
                    U[(i0 + di) * m + (j0 + dj)][col] += val
                # B: (dj, dk) -> global row j0+dj, col dk     -> V index (j0+dj)*p + dk
                for (dj, dk), val in v_t:
                    V[(j0 + dj) * p + dk][col] += val
                # C: (dk, di) -> global col dk, row i0+di     -> W index dk*n + (i0+di)
                for (dk, di), val in w_t:
                    W[dk * n + (i0 + di)][col] += val
                col += 1
    assert col == rank
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s

    # Guaranteed valid baseline.
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    if not _verify_exact(best["U"], best["V"], best["W"], n, m, p):
        # schoolbook must always verify; if not, still return it.
        return best

    # Candidate 1: block-Winograd (rank 7*p per 2x2 block).
    try:
        cand = _block_winograd(n, m, p)
        r = len(cand["U"][0])
        if r < best_rank and _verify_exact(cand["U"], cand["V"], cand["W"], n, m, p):
            best, best_rank = cand, r
    except Exception:
        pass

    # Candidate 2: full 4x4x9 Strassen (rank 49) -- attempt via recursive 2x2.
    if time.time() < deadline:
        try:
            cand4 = _strassen_4x4x4(n, m, p)
            r = len(cand4["U"][0])
            if r < best_rank and _verify_exact(cand4["U"], cand4["V"], cand4["W"], n, m, p):
                best, best_rank = cand4, r
        except Exception:
            pass

    return best


def _strassen_4x4x4(n, m, p):
    """Recursive Strassen: 4x4x4 -> 7 * (2x2x4) -> 7 * 7 * (1x1x4) = 196? No.
    Actually 4x4x4 Strassen = 7 * (2x2x2) = 7*7 = 49. But our p=9 not 4.
    For 4x4x9: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1.
    4x4x4 Strassen rank 49; 4x4x1 schoolbook rank 16 -> 49+49+16 = 114.
    """
    # Build 4x4x4 Strassen (rank 49) for the first 4 cols, then 4x4x1 schoolbook,
    # then another 4x4x4 for cols 4..7, then 4x4x1 for col 8.
    def _strassen_4x4x4_block(U, V, W, col_off, k_start):
        # 7 factors, each (2x2x2) -> 7 sub-factors of (1x1x2)
        # A 4x4, B 4x4 (rows 0..3, cols k_start..k_start+3), C 4x4 (cols k_start.., rows 0..3)
        # 2x2 blocks: A block (bi,bj) at rows 2bi.., cols 2bj..
        # B block (bj,bk) at rows 2bj.., cols k_start+2bk..
        # C block (bk,bi) at cols k_start+2bk.., rows 2bi..
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # This is a 2x2x2 Strassen with rank 7
                    i0 = 2 * bi
                    j0 = 2 * bj
                    k0 = k_start + 2 * bk
                    for (u_t, v_t, w_t) in _STRASSEN:
                        for (di, dj), val in u_t:
                            U[(i0 + di) * m + (j0 + dj)][col_off] += val
                        for (dj, dk), val in v_t:
                            V[(j0 + dj) * p + (k0 + dk)][col_off] += val
                        for (dk, di), val in w_t:
                            W[(k0 + dk) * n + (i0 + di)][col_off] += val
                        col_off += 1
        return col_off

    # total rank = 3 blocks of 4x4x4 (49 each) + 1 block of 4x4x1 (16)
    total = 49 + 49 + 16
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    col = 0
    col = _strassen_4x4x4_block(U, V, W, col, 0)   # cols 0..3
    col = _strassen_4x4x4_block(U, V, W, col, 4)   # cols 4..7
    # 4x4x1 schoolbook for col 8
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    assert col == total
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
