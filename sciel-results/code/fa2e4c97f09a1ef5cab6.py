"""Matrix multiplication 3x4x8 with rank-reduced decomposition.

Structural idea (from current program): a tiling of the 3x4x8 multiplication
tensor into small blocks (2x2x2 Strassen rank-7 for the first two output
rows, 1x2x2 schoolbook rank-4 for the third), assembled into global U, V, W.

Refinement machinery (from the inspiration): a verification-driven local
search that looks for lower-rank replacements of a 3x2x2 sub-block (the
third output row over a 2x2 window) using a smooth log-sum-exp surrogate of
the max-norm residual, alternating least squares, and integer/half-integer
rounding -- accepting a change ONLY if an exact integer verification passes.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).

    Order of A elements: a00, a01, a10, a11
    Order of B elements: b00, b01, b10, b11
    Order of C elements: c00, c01, c10, c11

    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0],    # b01
        [0, 0, 0, 1, 0, 0, 1],    # b10
        [1, 0, -1, 0, 1, 0, 1]    # b11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0]    # c11
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification.

    tiling: list of (btype, js, je, ks, ke). Each block spans
    j in [js, je), k in [ks, ke).
    """
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []

    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        elif btype == 'combined3':
            r = 6
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()

    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1

        if btype == 'strassen2':
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4

    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i in {0,1}, 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > tol:
                    return False
    return True


def _schoolbook_3x2x2():
    """Return U, V, W for 3x2x2 schoolbook decomposition (rank 6).

    A: 3x2 (6 entries), B: 2x2 (4 entries), C: 2x3 (6 entries).
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1}.
    """
    n, m, p = 3, 2, 2
    r = 6
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for i in range(n):
        for j in range(m):
            col = i * m + j
            U[i * m + j, col] = 1
            V[j * p + 0, col] = 1
            V[j * p + 1, col] = 1
            W[0 * n + i, col] = 1
            W[1 * n + i, col] = 1
    return U, V, W


def _verify_block(Ub, Vb, Wb, n, m, p, tol=1e-9):
    """Verify a standalone block decomposition (numpy arrays)."""
    r = Ub.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += Ub[i * m + j, c] * Vb[j * p + k, c] * Wb[k * n + i, c]
                if abs(val - 1) > tol:
                    return False
    return True


def _soft_max_norm(res, beta=20.0):
    """Smooth surrogate for max |res| using log-sum-exp of |res|^2."""
    a = np.square(res)
    m = a.max()
    if m == 0:
        return 0.0
    return (m + np.log(np.sum(np.exp(a - m))) / a.size) ** 0.5 / beta


def _round_half(x):
    """Round to nearest integer or half-integer."""
    return np.round(2.0 * x) / 2.0


def _search_3x2x2_low_rank(time_budget):
    """Search for a rank-5 (or lower) 3x2x2 decomposition.

    Uses a smooth log-sum-exp surrogate of the max-norm residual with
    alternating least squares and integer/half-integer rounding. Returns
    (U, V, W) as numpy arrays if a valid decomposition is found, else None.
    """
    start = time.time()
    n, m, p = 3, 2, 2
    # Target tensor T[i,j,k] = 1 for all (i,j,k).
    # Residual for a candidate (U,V,W): R[i,j,k] = sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1.
    best = None
    best_obj = np.inf

    # Try a few random initializations in the half-integer lattice.
    for trial in range(40):
        if time.time() - start > time_budget:
            break
        r = 5
        rng = np.random.RandomState(1000 + trial)
        U = _round_half(rng.randint(-1, 2, size=(n * m, r)))
        V = _round_half(rng.randint(-1, 2, size=(m * p, r)))
        W = _round_half(rng.randint(-1, 2, size=(p * n, r)))

        # A few alternating least-squares polish steps (continuous), then round.
        for it in range(20):
            # Update U
            for idx in range(n * m):
                i, j = divmod(idx, m)
                # R contribution: U[idx,c] * V[j,k,c] * W[k,i,c]
                # Build a matrix over c for each (j,k): coeff[c] = V[j,k,c]*W[k,i,c]
                # We want sum_c U[idx,c]*coeff_{j,k}[c] = 1 for all k.
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                U[idx, :] = sol
            # Update V
            for idx in range(m * p):
                j, k = divmod(idx, p)
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                V[idx, :] = sol
            # Update W
            for idx in range(p * n):
                k, i = divmod(idx, n)
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                W[idx, :] = sol

        # Round to half-integers and evaluate the smooth surrogate.
        Ur = _round_half(U)
        Vr = _round_half(V)
        Wr = _round_half(W)
        # Compute residual.
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += Ur[i * m + j, c] * Vr[j * p + k, c] * Wr[k * n + i, c]
                    res[i, j, k] = val - 1
        obj = _soft_max_norm(res)
        if obj < best_obj:
            best_obj = obj
            best = (Ur, Vr, Wr)
        # If already exact, stop early.
        if np.max(np.abs(res)) < 1e-9:
            return Ur, Vr, Wr

    if best is not None and _verify_block(best[0], best[1], best[2], n, m, p):
        return best
    return None


def _construct_3x4x8():
    """Construct the decomposition, attempting a rank reduction on i=2 blocks."""
    tiling = _default_tiling()
    return _build_decomposition(tiling)


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before the 150s limit

    result = _construct_3x4x8()
    n, m, p = DIMENSIONS

    # Refinement: try to find a rank-5 3x2x2 block and, if valid, rebuild
    # the i=2 row using it (8 blocks x rank 5 = 40 instead of 8 x 4 = 32...
    # note rank 5 > 4, so only accept if it actually lowers total rank).
    # A rank-5 3x2x2 block is NOT an improvement over the rank-4 schoolbook
    # 1x2x2 for a single row. The genuine win would be a rank < 4 block or a
    # cross-block sharing. We attempt the search but only accept an exact,
    # strictly-lower-rank result; otherwise keep the current tiling.
    time_left = deadline - time.time()
    if time_left > 30:
        block = _search_3x2x2_low_rank(time_budget=min(60.0, time_left - 10))
        if block is not None:
            # Build a tiling that uses this rank-5 block for the i=2 row.
            # Total rank would be 8*7 + 8*5 = 96 > 88, so do NOT use it.
            # (Kept for correctness of the machinery; not an improvement.)
            pass

    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}

    return result
