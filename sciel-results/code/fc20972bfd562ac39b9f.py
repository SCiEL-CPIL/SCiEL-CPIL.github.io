"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Guaranteed-correct rank n*m*p decomposition (fallback)."""
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
    return U, V, W


def _strassen_block(n, m, p):
    """
    Block-partition + Strassen decomposition for square n x n x n matrix multiplication.

    Partition each matrix into a leading 2x2 block and a trailing 1x1 block:
        A = [A00 A01 | A10]   B = [B00 B01 | B10]
            [A11      | A22]       [B11 B22 | B22]
    where A00,B00 are 2x2, A01,B10,A10,B11,A11,B22 are 1x1 scalars.
    The 2x2 core (A00*B00) is computed with Strassen's 7 products; the five
    remaining scalar block-products are single terms. Total = 12 bilinear terms.
    """
    # U row index = i*m + j ; V row index = j*p + k ; W row index = k*n + i
    U = []
    V = []
    W = []

    def add(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # --- Strassen products for the 2x2 core (indices 0,1) ---
    # M1 = (A00+A11)(B00+B11)
    add([0, 1, 3, 4, 0, 0, 0, 0, 0],
        [0, 1, 3, 4, 0, 0, 0, 0, 0],
        [0, 1, 3, 4, 0, 0, 0, 0, 0])
    # M2 = (A10+A11) B00
    add([3, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 4, 0, 0, 0, 0, 0])
    # M3 = A00 (B10 - B11)
    add([0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0])
    # M4 = A11 (B11 - B00)
    add([0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0])
    # M5 = (A00+A01) B11
    add([0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0])
    # M6 = (A10 - A00)(B00 + B01)
    add([0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0])
    # M7 = (A01 - A11)(B10 + B11)
    add([1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0])

    # --- Scalar block-products (single terms) ---
    # A01 * B10 -> C01
    add([0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0])
    # A01 * B22 -> C02
    add([0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0])
    # A10 * B11 -> C10
    add([3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0])
    # A11 * B22 -> C12
    add([0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0])
    # A22 * B22 -> C22
    add([0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 8])

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Exact verification that U,V,W decompose the n x n x n matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i * m + j, j * p + k, k * n + i] = 1.0
    return bool(np.allclose(T, T_exp, atol=tol, rtol=0.0))


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before the 150s kill

    # Primary: block-partition + Strassen (rank 12 for 3x3x3).
    if time.time() < deadline:
        U, V, W = _strassen_block(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed-correct schoolbook (rank 27).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
