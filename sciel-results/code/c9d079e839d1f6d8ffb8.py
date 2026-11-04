"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_strassen3(rank3):
    """Build a valid rank-`rank3` decomposition for 3x3 matrix multiplication.

    Returns U,V,W as numpy integer arrays of shape (9,rank3) with the
    official trace convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c].
    Uses a 2x2 Strassen block (7 products) plus 16 direct products for the
    remaining (off-diagonal) block entries.
    """
    s3 = 3
    U = np.zeros((s3 * s3, rank3), dtype=object)
    V = np.zeros((s3 * s3, rank3), dtype=object)
    W = np.zeros((s3 * s3, rank3), dtype=object)

    col = 0
    # 7 Strassen products on the top-left 2x2 block (rows/cols 0,1 of A,B,C).
    # Strassen:
    #   P1 = (A00+A11)(B00+B11)
    #   P2 = (A10+A11)B00
    #   P3 = A00(B01-B11)
    #   P4 = A11(B10-B00)
    #   P5 = (A00+A01)B11
    #   P6 = (A10-A00)(B00+B01)
    #   P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    def add_U(i, j, val):
        U[i * s3 + j, col] += val
    def add_V(j, k, val):
        V[j * s3 + k, col] += val
    def add_W(k, i, val):
        W[k * s3 + i, col] += val

    # P1 = (A00+A11)(B00+B11)
    add_U(0, 0, 1); add_U(1, 1, 1)
    add_V(0, 0, 1); add_V(1, 1, 1)
    add_W(0, 0, 1); add_W(1, 1, -1); add_W(1, 0, 1); add_W(0, 1, 1)
    col += 1
    # P2 = (A10+A11)B00
    add_U(1, 0, 1); add_U(1, 1, 1)
    add_V(0, 0, 1)
    add_W(0, 1, 1); add_W(1, 1, -1)
    col += 1
    # P3 = A00(B01-B11)
    add_U(0, 0, 1)
    add_V(0, 1, 1); add_V(1, 1, -1)
    add_W(1, 0, 1); add_W(1, 1, 1)
    col += 1
    # P4 = A11(B10-B00)
    add_U(1, 1, 1)
    add_V(1, 0, 1); add_V(0, 0, -1)
    add_W(0, 0, 1); add_W(1, 0, 1); add_W(0, 1, 1)
    col += 1
    # P5 = (A00+A01)B11
    add_U(0, 0, 1); add_U(0, 1, 1)
    add_V(1, 1, 1)
    add_W(1, 0, -1); add_W(1, 1, 1)
    col += 1
    # P6 = (A10-A00)(B00+B01)
    add_U(1, 0, 1); add_U(0, 0, -1)
    add_V(0, 0, 1); add_V(0, 1, 1)
    add_W(1, 1, 1)
    col += 1
    # P7 = (A01-A11)(B10+B11)
    add_U(0, 1, 1); add_U(1, 1, -1)
    add_V(1, 0, 1); add_V(1, 1, 1)
    add_W(0, 0, 1); add_W(0, 1, -1)
    col += 1

    # Direct products for the 5 off-diagonal entries of C (block indices
    # involving the third row/col): C02, C12, C20, C21, C22 -> 3+3+3+3+3=15,
    # but C02=C03? No: 3x3 has entries (0,2),(1,2),(2,0),(2,1),(2,2) = 5 entries,
    # each needs 3 products = 15. Total = 7 + 15 = 22. We need rank 23, so
    # keep one extra slot unused (or add a harmless zero). We'll use 22 and
    # leave column 22 zero to keep shape (9,23); zeros are harmless.
    for ci in range(3):
        for cj in range(3):
            if (ci < 2 and cj < 2):
                continue  # handled by Strassen
            for k in range(3):
                add_U(ci, k, 1)
                add_V(k, cj, 1)
                add_W(cj, ci, 1)
                col += 1
    # col should be 22; remaining columns (22) stay zero.
    assert col <= rank3, col
    return U, V, W


def _to_lists(U, V, W):
    return (U.tolist(), V.tolist(), W.tolist())


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 iff i==k and j==n? ...
    The tensor T[i,j,k] = 1 if (i,k) is the matrix-multiplication pairing with
    the trace convention: T[i*m+j, j*p+k, k*n+i] = 1 exactly when the
    (i,k) output element equals sum_j A[i,j]B[j,k]. The scalar tensor is
    T[i,j,k] = delta_{i,k}? No. For matrix mult C=A*B, C[i,k]=sum_j A[i,j]B[j,k].
    The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k (each
    contributes A[i,j]*B[j,k] to C[i,k]). So the decomposition must satisfy:
    for every (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    ok = False
                    return False, (i, j, k, s)
    return ok, None


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    n, m, p = DIMENSIONS
    try:
        U3, V3, W3 = _build_strassen3(23)
        # Build 6x6 by two 3x3 diagonal blocks.
        # Block 1 uses rows/cols 0..2 of the 6x6; Block 2 uses rows/cols 3..5.
        rank = 46
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        for block in range(2):
            base = block * 3
            off = block * 23
            for i in range(3):
                for j in range(3):
                    U[(base + i) * m + (base + j), off] = U3[i * 3 + j, 0]
                    U[(base + i) * m + (base + j), off + 1] = U3[i * 3 + j, 1]
                    U[(base + i) * m + (base + j), off + 2] = U3[i * 3 + j, 2]
                    U[(base + i) * m + (base + j), off + 3] = U3[i * 3 + j, 3]
                    U[(base + i) * m + (base + j), off + 4] = U3[i * 3 + j, 4]
                    U[(base + i) * m + (base + j), off + 5] = U3[i * 3 + j, 5]
                    U[(base + i) * m + (base + j), off + 6] = U3[i * 3 + j, 6]
                    for c in range(7, 23):
                        U[(base + i) * m + (base + j), off + c] = U3[i * 3 + j, c]
            for j in range(3):
                for k in range(3):
                    V[(base + j) * p + (base + k), off] = V3[j * 3 + k, 0]
                    for c in range(1, 23):
                        V[(base + j) * p + (base + k), off + c] = V3[j * 3 + k, c]
            for k in range(3):
                for i in range(3):
                    W[(base + k) * n + (base + i), off] = W3[k * 3 + i, 0]
                    for c in range(1, 23):
                        W[(base + k) * n + (base + i), off + c] = W3[k * 3 + i, c]
        # Verify exactness.
        ok, info = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Fallback: schoolbook rank 216.
    except Exception:
        pass
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
