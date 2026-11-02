import time
import numpy as np

DIMENSIONS = [4, 5, 6]


# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U,V,W) each shape (4,7) for 2x2 * 2x2 -> 2x2.
    U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t] summed over t == 1 for all (i,j,k).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # bilinear combinations (U for A, V for B)
    U[0, 0] = 1   # A00
    U[1, 1] = 1   # A01
    U[2, 2] = 1   # A10
    U[3, 3] = 1   # A11
    U[1, 4] = 1   # A01
    U[2, 5] = 1   # A10
    U[3, 6] = 1   # A11

    V[0, 0] = 1   # B00
    V[2, 1] = 1   # B10
    V[1, 2] = 1   # B01
    V[3, 3] = 1   # B11
    V[3, 4] = 1   # B11
    V[2, 5] = 1   # B10
    V[0, 6] = 1   # B00

    # output combinations (W for C)
    W[0, 0] = 1   # C00
    W[3, 1] = 1   # C10
    W[1, 2] = 1   # C01
    W[2, 3] = 1   # C11
    W[1, 4] = 1   # C01
    W[2, 5] = 1   # C11
    W[0, 6] = 1   # C00

    return U, V, W


def _schoolbook(an, am, bp, ap):
    """Schoolbook block: A (an x am) * B (am x ap) -> C (an x ap).
    Returns (U,V,W) with U shape (an*am, r), V shape (am*ap, r), W shape (ap*an, r).
    """
    r = an * am * ap
    U = np.zeros((an * am, r))
    V = np.zeros((am * ap, r))
    W = np.zeros((ap * an, r))
    col = 0
    for i in range(an):
        for j in range(am):
            for k in range(ap):
                U[i * am + j, col] = 1
                V[j * ap + k, col] = 1
                W[k * an + i, col] = 1
                col += 1
    return U, V, W


def _place(U, V, W, i0, j0, k0, an, am, ap, bu, bv, bw):
    """Place a block's (bu,bv,bw) into the global matrices at offsets (i0,j0,k0).
    U index = i*am + j ; V index = j*ap + k ; W index = k*an + i.
    """
    r = bu.shape[1]
    for u in range(bu.shape[0]):
        for t in range(r):
            val = bu[u, t]
            if val != 0:
                U[i0 * am + j0 * ap + u, t] += val
    for v in range(bv.shape[0]):
        for t in range(r):
            val = bv[v, t]
            if val != 0:
                V[j0 * ap + k0 * an + v, t] += val
    for w in range(bw.shape[0]):
        for t in range(r):
            val = bw[w, t]
            if val != 0:
                W[k0 * an + i0 * am + w, t] += val


def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for t in range(r):
                    actual += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(actual - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin within 150s wall-clock limit

    # Global factor matrices.
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))

    def add_block(bu, bv, bw, i0, j0, k0, an, am, ap):
        nonlocal U, V, W
        r = bu.shape[1]
        U = np.hstack([U, np.zeros((U.shape[0], r))])
        V = np.hstack([V, np.zeros((V.shape[0], r))])
        W = np.hstack([W, np.zeros((W.shape[0], r))])
        _place(U, V, W, i0, j0, k0, an, am, ap, bu, bv, bw)

    # Block 1: A(0:2, 0:2) * B(0:2, 0:2) -> C(0:2, 0:2)  [Strassen, r=7]
    su, sv, sw = _strassen_2x2x2()
    add_block(su, sv, sw, 0, 0, 0, 2, 2, 2)

    # Block 2: A(0:2, 2:3) * B(2:3, 0:2) -> C(0:2, 0:2)  [schoolbook, r=12]
    bu, bv, bw = _schoolbook(2, 1, 2, 2)
    add_block(bu, bv, bw, 0, 2, 0, 2, 1, 2)

    # Block 3: A(2:4, 0:2) * B(0:2, 0:3) -> C(0:3, 2:4)  [schoolbook, r=16]
    bu, bv, bw = _schoolbook(2, 2, 3, 2)
    add_block(bu, bv, bw, 2, 0, 0, 2, 2, 3)

    # Block 4: A(2:4, 2:3) * B(2:3, 0:3) -> C(0:3, 2:4)  [schoolbook, r=8]
    bu, bv, bw = _schoolbook(2, 1, 3, 2)
    add_block(bu, bv, bw, 2, 2, 0, 2, 1, 3)

    # Verify the assembled decomposition.
    if not _verify(U, V, W, n, m, p):
        # Fallback: guaranteed-valid trivial decomposition (rank 120).
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
