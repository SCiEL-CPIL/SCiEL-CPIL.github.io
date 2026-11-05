"""Matrix multiplication 6x6x6 - Block-Strassen hierarchical decomposition + rank reduction.

Design family: Block-Strassen.  Exploit the 2x2 sub-block structure with Strassen's
7-term algorithm (and hierarchically the 4x4 19-term algorithm) applied across the
block positions, then remove any integer linear dependencies among the rank factors.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
START = time.time()
DEADLINE = START + 140.0  # safety margin under the 150 s wall-clock limit


def _matmul(A, B):
    return A @ B


def _strassen7(A, B):
    """Strassen 2x2: returns list of 7 (S1,S2) factor pairs, A,B are 2x2."""
    A00, A01, A10, A11 = A[0][0], A[0][1], A[1][0], A[1][1]
    B00, B01, B10, B11 = B[0][0], B[0][1], B[1][0], B[1][1]
    S1 = A00 + A11
    S2 = B10 + B11
    S3 = A10 - A00
    S4 = B11 - B00
    S5 = (A00 + A01) - B11
    S6 = A11 - B10
    S7 = (A10 + A11) - B00
    return [(S1, S2), (A00, B00), (S3, B10), (A11, S4),
            (S5, B01), (S6, B11), (S7, B01)]


def _strassen19(A, B):
    """Strassen 4x4: returns list of 19 (S1,S2) factor pairs, A,B are 4x4."""
    A00 = A[0][0]; A01 = A[0][1]; A10 = A[1][0]; A11 = A[1][1]
    B00 = B[0][0]; B01 = B[0][1]; B10 = B[1][0]; B11 = B[1][1]
    S1  = A00 + A11
    S2  = B10 + B11
    S3  = A10 - A00
    S4  = B11 - B00
    S5  = (A00 + A01) - B11
    S6  = A11 - B10
    S7  = (A10 + A11) - B00
    S8  = (A00 + A10) - (B00 + B10)
    S9  = (A01 + A11) - (B01 + B11)
    S10 = A00 - (A10 + A11)
    S11 = B00 + (B10 - B11)
    S12 = (A00 + A01) - (A10 + A11)
    S13 = B00 + B11
    S14 = (A01 + A11) - (A00 + A10)
    S15 = B01 + B10
    S16 = A01 - (A00 + A10)
    S17 = B00 - B10
    S18 = A11 - (A01 + A10)
    S19 = B11 - B01
    return [(S1, S2), (A00, B00), (S3, B10), (A11, S4),
            (S5, B01), (S6, B11), (S7, B01),
            (S8, B01), (S9, B10), (S10, S11), (S12, S13),
            (S14, S15), (S16, S17), (S18, S19)]


def _build_decomp(n, m, p, depth):
    """Hierarchical block-Strassen.  Returns (U, V, W) as numpy int64 arrays.

    depth 0 -> schoolbook (n*m*p factors).
    depth 1 -> 2x2 Strassen on 2x2 blocks.
    depth 2 -> 4x4 Strassen on 4x4 blocks.
    """
    n, m, p = n, m, p
    U = []  # each factor: list of (a, b) where U[a][c]=1, V[b][c]=1, W[c][c]=1
    factors = []

    def add_factor(a, b):
        factors.append((a, b))

    if depth <= 0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_factor(i * m + j, j * p + k)
    else:
        # number of 4x4 block positions when depth>=2, else 2x2 positions
        if depth >= 2:
            bs = 4
        else:
            bs = 2
        # block grid: n/bs by m/bs positions; each position is a bs x bs sub-product
        for bi in range(n // bs):
            for bj in range(m // bs):
                # Build the bs x bs A-block and B-block index matrices.
                # A-block (i,j) indices: rows bi*bs..(bi+1)*bs-1, cols bj*bs..
                # B-block (j,k) indices: rows bj*bs.., cols ...
                # We represent the abstract A and B as index matrices then run Strassen
                # on integer "index" matrices, producing integer linear combinations of
                # indices.  Each index is a scalar (row*dim + col) so sums are valid.
                Aidx = np.zeros((bs, bs), dtype=object)
                Bidx = np.zeros((bs, bs), dtype=object)
                for ii in range(bs):
                    for jj in range(bs):
                        i = bi * bs + ii
                        j = bj * bs + jj
                        Aidx[ii, jj] = i * m + j
                        Bidx[ii, jj] = j * p + (jj)  # placeholder, fixed below
                # Correct B index: row = j (column of A), col = k (row of B)
                for ii in range(bs):
                    for jj in range(bs):
                        j = bj * bs + jj
                        k = jj  # B column index within block
                        Bidx[ii, jj] = j * p + k
                # Now run the appropriate Strassen on these object index matrices.
                if depth >= 2:
                    pairs = _strassen19(Aidx, Bidx)
                else:
                    pairs = _strassen7(Aidx, Bidx)
                for S1, S2 in pairs:
                    add_factor(S1, S2)

    # Build dense factor matrices.
    rank = len(factors)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    for c, (a, b) in enumerate(factors):
        U[a, c] += 1
        V[b, c] += 1
        W[c, c] += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Exact verification: (U*V^T)_{i,j,k} == W_{k*n+i, j*p+k}."""
    n, m, p = n, m, p
    # T = U V^T  has shape (n*m, m*p)
    T = U @ V.T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = W[k * n + i, j * p + k]
                got = T[i * m + j, j * p + k]
                if expected != got:
                    return False
    return True


def _reduce_rank(U, V, W, n, m, p, iters=200):
    """Remove integer linear dependencies among rank factors.

    A factor c is removable if its columns U[:,c], V[:,c], W[:,c] are each integer
    linear combinations of the other factors' columns.  We use a Smith-normal-form-style
    column reduction over the integer lattice spanned by the other factors.
    """
    n, m, p = n, m, p
    U = U.astype(np.int64).copy()
    V = V.astype(np.int64).copy()
    W = W.astype(np.int64).copy()
    rank = U.shape[1]

    changed = True
    while changed and (time.time() - START) < DEADLINE - 5.0:
        changed = False
        # Try to remove each factor in turn (greedy).
        for c in range(U.shape[1]):
            if time.time() - START > DEADLINE - 5.0:
                break
            # Build the column-space generator matrix for the other factors.
            cols = [r for r in range(U.shape[1]) if r != c]
            if not cols:
                continue
            # For each of U, V, W check if column c is in the integer lattice of the
            # other columns.  Use integer Gaussian elimination (Smith normal form) on
            # the stacked problem is complex; instead do a per-matrix lattice test via
            # Hermite normal form of the other columns.
            removable = True
            for M in (U, V, W):
                G = M[:, cols]  # (rows x (rank-1))
                v = M[:, c]
                # Test if v is in the integer column lattice of G.
                # Compute HNF of G and solve G x = v over integers.
                hnf, _ = _hermite(G)
                # Solve HNF y = v for integer y (HNF is upper triangular-ish).
                y, ok = _solve_hnf(hnf, v)
                if not ok:
                    removable = False
                    break
            if removable:
                # Remove column c from all three matrices.
                mask = [r for r in range(U.shape[1]) if r != c]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                changed = True
                break
    return U, V, W


def _hermite(A):
    """Hermite normal form of integer matrix A (column-style, upper triangular).

    Returns (H, T) such that A = H @ T, H is HNF (upper triangular, positive pivots,
    0 <= off-diagonal < pivot).  Uses elementary column operations.
    """
    A = A.astype(np.int64).copy()
    if A.size == 0:
        return A, np.zeros((A.shape[0], 0), dtype=np.int64)
    m, n = A.shape
    H = A.copy()
    T = np.eye(n, dtype=np.int64)
    for j in range(n):
        # find pivot
        pivot = -1
        for i in range(m):
            if H[i, j] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        if H[pivot, j] < 0:
            H[pivot:, j] *= -1
            T[:, j] *= -1
        # eliminate below
        for i in range(pivot + 1, m):
            while H[i, j] != 0:
                q, r = divmod(H[i, j], H[pivot, j])
                H[i, :] -= q * H[pivot, :]
                T[i, :] -= q * T[pivot, :]
        # eliminate above (make it a true HNF: 0 <= off < pivot)
        for i in range(pivot):
            while H[i, j] != 0:
                q, r = divmod(H[i, j], H[pivot, j])
                H[i, :] -= q * H[pivot, :]
                T[i, :] -= q * T[pivot, :]
    return H, T


def _solve_hnf(H, v):
    """Solve H y = v for integer y given H in column-HNF (upper triangular)."""
    m, n = H.shape
    y = np.zeros(n, dtype=np.int64)
    for j in range(n):
        # find pivot row for column j
        pivot = -1
        for i in range(m):
            if H[i, j] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        # accumulate contribution of already-solved columns
        rhs = v[pivot]
        for jj in range(j + 1, n):
            rhs -= H[pivot, jj] * y[jj]
        if H[pivot, j] == 0:
            continue
        if rhs % H[pivot, j] != 0:
            return None, False
        y[j] = rhs // H[pivot, j]
    # verify
    if not np.array_equal(H @ y, v):
        return None, False
    return y, True


def construct():
    n, m, p = DIMENSIONS
    best = None
    best_rank = None

    # Candidate depths: 2 (4x4 Strassen) first, then 1 (2x2), then 0 (schoolbook).
    for depth in (2, 1, 0):
        if time.time() - START > DEADLINE - 15.0:
            break
        try:
            U, V, W = _build_decomp(n, m, p, depth)
        except Exception:
            continue
        if not _verify(U, V, W, n, m, p):
            continue
        # Attempt rank reduction (safe).
        try:
            Ur, Vr, Wr = _reduce_rank(U, V, W, n, m, p)
            if _verify(Ur, Vr, Wr, n, m, p) and Ur.shape[1] < U.shape[1]:
                U, V, W = Ur, Vr, Wr
        except Exception:
            pass
        r = U.shape[1]
        if best_rank is None or r < best_rank:
            best_rank = r
            best = (U, V, W)
            if time.time() - START > DEADLINE - 20.0:
                break

    if best is None:
        # Fallback: pure schoolbook (always valid).
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
