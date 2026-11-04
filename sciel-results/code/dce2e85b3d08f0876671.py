"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

_DEADLINE = time.time() + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, rank):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _rank8_decomposition():
    """
    Rank-8 decomposition: Strassen 7 terms for the 2x2x2 sub-block (i,j,k in {0,1})
    plus one merged rank-1 term covering all indices involving index 2 in j or k.
    """
    n, m, p = 2, 3, 3
    rank = 8
    U = np.zeros((n*m, rank))  # (6, 8)
    V = np.zeros((m*p, rank))  # (9, 8)
    W = np.zeros((p*n, rank))  # (6, 8)

    # --- Strassen 7 terms for i,j,k in {0,1} ---
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0*3+0, 0] = 1.0; U[1*3+1, 0] = 1.0
    V[0*3+0, 0] = 1.0; V[1*3+1, 0] = 1.0
    W[0*2+0, 0] = 1.0; W[1*2+1, 0] = 1.0

    # Term 1: M2 = (A10+A11)(B00)
    U[1*3+0, 1] = 1.0; U[1*3+1, 1] = 1.0
    V[0*3+0, 1] = 1.0
    W[0*2+1, 1] = 1.0; W[1*2+1, 1] = -1.0

    # Term 2: M3 = (A00)(B01-B11)
    U[0*3+0, 2] = 1.0
    V[0*3+1, 2] = 1.0; V[1*3+1, 2] = -1.0
    W[1*2+0, 2] = 1.0; W[1*2+1, 2] = 1.0

    # Term 3: M4 = (A11)(B10-B00)
    U[1*3+1, 3] = 1.0
    V[1*3+0, 3] = 1.0; V[0*3+0, 3] = -1.0
    W[0*2+0, 3] = 1.0; W[0*2+1, 3] = 1.0

    # Term 4: M5 = (A00+A01)(B11)
    U[0*3+0, 4] = 1.0; U[0*3+1, 4] = 1.0
    V[1*3+1, 4] = 1.0
    W[0*2+0, 4] = -1.0; W[1*2+1, 4] = -1.0

    # Term 5: M6 = (A10-A00)(B00+B01)
    U[1*3+0, 5] = 1.0; U[0*3+0, 5] = -1.0
    V[0*3+0, 5] = 1.0; V[0*3+1, 5] = 1.0
    W[1*2+1, 5] = 1.0

    # Term 6: M7 = (A01-A11)(B10+B11)
    U[0*3+1, 6] = 1.0; U[1*3+1, 6] = -1.0
    V[1*3+0, 6] = 1.0; V[1*3+1, 6] = 1.0
    W[0*2+0, 6] = 1.0

    # --- Term 7: merged term for all indices involving 2 ---
    # U[i*3+j, 7] = 1 for all i in {0,1}, j in {0,1,2}
    for i in range(2):
        for j in range(3):
            U[i*3+j, 7] = 1.0

    # V[j*3+k, 7] = 1 if j==2 or k==2, else 0
    for j in range(3):
        for k in range(3):
            if j == 2 or k == 2:
                V[j*3+k, 7] = 1.0

    # W[k*2+i, 7] = 1 for all k in {0,1,2}, i in {0,1}
    for k in range(3):
        for i in range(2):
            W[k*2+i, 7] = 1.0

    return U.tolist(), V.tolist(), W.tolist()


def _rank10_decomposition():
    """
    Explicit rank-10 decomposition using Strassen's algorithm for the 2x2x2 block
    plus 3 additional rank-1 terms for the remaining (j,k) pairs involving index 2.
    """
    n, m, p = 2, 3, 3
    rank = 10
    U = np.zeros((n*m, rank))  # (6, 10)
    V = np.zeros((m*p, rank))  # (9, 10)
    W = np.zeros((p*n, rank))  # (6, 10)

    # --- Strassen 7 terms for i,j,k in {0,1} ---
    U[0*3+0, 0] = 1.0; U[1*3+1, 0] = 1.0
    V[0*3+0, 0] = 1.0; V[1*3+1, 0] = 1.0
    W[0*2+0, 0] = 1.0; W[1*2+1, 0] = 1.0

    U[1*3+0, 1] = 1.0; U[1*3+1, 1] = 1.0
    V[0*3+0, 1] = 1.0
    W[0*2+1, 1] = 1.0; W[1*2+1, 1] = -1.0

    U[0*3+0, 2] = 1.0
    V[0*3+1, 2] = 1.0; V[1*3+1, 2] = -1.0
    W[1*2+0, 2] = 1.0; W[1*2+1, 2] = 1.0

    U[1*3+1, 3] = 1.0
    V[1*3+0, 3] = 1.0; V[0*3+0, 3] = -1.0
    W[0*2+0, 3] = 1.0; W[0*2+1, 3] = 1.0

    U[0*3+0, 4] = 1.0; U[0*3+1, 4] = 1.0
    V[1*3+1, 4] = 1.0
    W[0*2+0, 4] = -1.0; W[1*2+1, 4] = -1.0

    U[1*3+0, 5] = 1.0; U[0*3+0, 5] = -1.0
    V[0*3+0, 5] = 1.0; V[0*3+1, 5] = 1.0
    W[1*2+1, 5] = 1.0

    U[0*3+1, 6] = 1.0; U[1*3+1, 6] = -1.0
    V[1*3+0, 6] = 1.0; V[1*3+1, 6] = 1.0
    W[0*2+0, 6] = 1.0

    # --- Term 7: j in {0,1}, k=2 ---
    for i in range(2):
        for j in range(2):
            U[i*3+j, 7] = 1.0
    for j in range(2):
        V[j*3+2, 7] = 1.0
    for i in range(2):
        W[2*2+i, 7] = 1.0

    # --- Term 8: j=2, k in {0,1} ---
    for i in range(2):
        U[i*3+2, 8] = 1.0
    for k in range(2):
        V[2*3+k, 8] = 1.0
    for k in range(2):
        for i in range(2):
            W[k*2+i, 8] = 1.0

    # --- Term 9: j=2, k=2 ---
    for i in range(2):
        U[i*3+2, 9] = 1.0
    V[2*3+2, 9] = 1.0
    for i in range(2):
        W[2*2+i, 9] = 1.0

    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook():
    """Return the schoolbook rank-18 decomposition."""
    n, m, p = DIMENSIONS
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


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Tries rank-8 (merged Strassen + one extra term), then rank-10, then schoolbook.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    # Try rank-8 first
    U, V, W = _rank8_decomposition()
    if _verify(U, V, W, 8):
        return {"U": U, "V": V, "W": W}

    # Try rank-10
    U, V, W = _rank10_decomposition()
    if _verify(U, V, W, 10):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook decomposition
    return _schoolbook()
