"""Matrix multiplication 4x4x6 via recursive Strassen (4x4x4, rank 49) +
block Strassen (4x4x2, rank 28) = rank 77, plus a rank-reduction polish pass."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2():
    """2x2x2 Strassen factors. U:(4,7), V:(4,7), W:(4,7). Index W[k*2+i]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    prods = [
        ([(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)]),
        ([(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)], [(1, 0, 1.0), (1, 1, -1.0)]),
        ([(0, 0, 1.0)], [(0, 1, 1.0), (1, 1, -1.0)], [(0, 1, 1.0), (1, 1, 1.0)]),
        ([(1, 1, 1.0)], [(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (1, 0, 1.0)]),
        ([(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)], [(0, 0, -1.0), (0, 1, 1.0)]),
        ([(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)]),
        ([(0, 1, 1.0), (1, 1, -1.0)], [(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)]),
    ]
    for c, (ub, vb, wb) in enumerate(prods):
        for (i, j, cv) in ub:
            U[i * 2 + j, c] += cv
        for (j, k, cv) in vb:
            V[j * 2 + k, c] += cv
        for (i, k, cv) in wb:
            W[k * 2 + i, c] += cv
    return U, V, W


def _build_444():
    """4x4x4 recursive Strassen via Kronecker of 2x2 Strassen. Rank 49.
    U:(16,49), V:(16,49), W:(16,49)."""
    Us, Vs, Ws = _strassen_2x2()
    r = 49
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            U[i * 4 + j, col] = Us[i1 * 2 + j1, r1] * Us[i2 * 2 + j2, r2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2 * j1 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            V[j * 4 + k, col] = Vs[j1 * 2 + k1, r1] * Vs[j2 * 2 + k2, r2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2 * k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            W[k * 4 + i, col] = Ws[k1 * 2 + i1, r1] * Ws[k2 * 2 + i2, r2]
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    Ur = np.asarray(U, dtype=float).reshape(n, m, rank)
    Vr = np.asarray(V, dtype=float).reshape(m, p, rank)
    Wr = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))


def _to_list(mat):
    out = []
    for row in mat:
        nr = []
        for x in row:
            xr = round(x)
            if abs(x - xr) < 1e-7:
                nr.append(int(xr))
            else:
                nh = round(2 * x) / 2.0
                if abs(x - nh) < 1e-7:
                    nr.append(nh)
                else:
                    nr.append(float(x))
        out.append(nr)
    return out


def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    Us, Vs, Ws = _strassen_2x2()
    U444, V444, W444 = _build_444()

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = np.zeros((n * m, total_rank))  # 16 x 77
    V = np.zeros((m * p, total_rank))  # 24 x 77
    W = np.zeros((p * n, total_rank))  # 24 x 77

    # Part 1: 4x4x4 recursive Strassen for B columns 0-3 -> C columns 0-3
    for idx in range(16):
        U[idx, :rank_444] = U444[idx]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, :rank_444] = V444[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :rank_444] = W444[k * 4 + i]

    # Part 2: 4x4x2 block Strassen for B columns 4-5 -> C columns 4-5
    # C_top = A_top[:,0:2]*B_top + A_top[:,2:4]*B_bottom
    # C_bottom = A_bottom[:,0:2]*B_top + A_bottom[:,2:4]*B_bottom
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]
    for bidx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + bidx * 7 + s
            for i in range(2):
                for j in range(2):
                    v = Us[i * 2 + j, s]
                    if v:
                        U[(ar + i) * 4 + (ac + j), col] += v
            for j in range(2):
                for k in range(2):
                    v = Vs[j * 2 + k, s]
                    if v:
                        V[(br + j) * 6 + (bc + k), col] += v
            for k in range(2):
                for i in range(2):
                    v = Ws[k * 2 + i, s]
                    if v:
                        W[(cc + k) * 4 + (cr + i), col] += v

    best = (U.copy(), V.copy(), W.copy(), total_rank)
    err = _verify(U, V, W, n, m, p, total_rank)
    if err > 1e-6:
        # Fallback: trivial rank n*m*p
        rank = n * m * p
        Uf = np.zeros((n * m, rank))
        Vf = np.zeros((m * p, rank))
        Wf = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, col] = 1
                    Vf[j * p + k, col] = 1
                    Wf[k * n + i, col] = 1
                    col += 1
        return {"U": _to_list(Uf), "V": _to_list(Vf), "W": _to_list(Wf)}

    # ---- Rank-reduction polish: SVD-based low-rank exact search ----
    # If the 77-term decomposition is not minimal, the tensor may be representable
    # at lower rank. Try to find the smallest r such that U,V,W (r columns) still
    # exactly reproduce the multiplication tensor, with integer/half-integer entries.
    def try_rank(r):
        """Attempt to find an exact r-rank decomposition. Return (U,V,W) or None."""
        # Use the current 77-term decomposition's tensor to guide SVD.
        # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
        # We want to find U,V,W of rank r reproducing T.
        # This is a non-convex problem; use the existing 77-term factors projected
        # onto a low-rank subspace via SVD of the factor matrices.
        # Strategy: SVD of U (16x77) -> keep top r left singular vectors, etc.
        # But the three factors must be consistent, so we iterate.
        # Simpler: try random projections and check exactness.
        # Given time budget, do a few SVD-based attempts.
        Ucur = U.copy()
        Vcur = V.copy()
        Wcur = W.copy()
        for _ in range(3):
            # Project U onto top-r subspace
            try:
                Uu, Usig, Uvt = np.linalg.svd(Ucur, full_matrices=False)
                Vv, Vsig, Vvt = np.linalg.svd(Vcur, full_matrices=False)
                Ww, Wsig, Wvt = np.linalg.svd(Wcur, full_matrices=False)
            except np.linalg.LinAlgError:
                return None
            # Reconstruct with rank r
            Ur = Uu[:, :r] * Usig[:r]
            Vr = Vv[:, :r] * Vsig[:r]
            Wr = Ww[:, :r] * Wsig[:r]
            # Check exactness
            e = _verify(Ur, Vr, Wr, n, m, p, r)
            if e < 1e-6:
                # Snap to integer/half-integer
                Ur = np.round(2 * Ur) / 2.0
                Vr = np.round(2 * Vr) / 2.0
                Wr = np.round(2 * Wr) / 2.0
                e2 = _verify(Ur, Vr, Wr, n, m, p, r)
                if e2 < 1e-6:
                    return (Ur, Vr, Wr)
            # If not exact, the SVD projection didn't work; break early.
            if e > 1e-3:
                return None
        return None

    # Try ranks below 77, descending
    for r in range(total_rank - 1, 0, -1):
        if time.time() > deadline - 20.0:
            break
        res = try_rank(r)
        if res is not None:
            U, V, W, total_rank = res[0], res[1], res[2], r
            best = (U.copy(), V.copy(), W.copy(), r)
            break

    U, V, W, total_rank = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
