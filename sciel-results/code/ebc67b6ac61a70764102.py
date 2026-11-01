import numpy as np
import time

DIMENSIONS = [2, 2, 15]
_DEADLINE_SECONDS = 120.0

def _build_strassen_2x2x2():
    """
    Strassen rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2.
    Returns U (4x7), V (4x7), W (4x7) with the LOCAL k-index (0,1).
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1; W[3, 1] = -1
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p (fallback)."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

    Strategy (Coppersmith-Winograd-style blocking):
      - Split the 15 output columns into seven 2-column blocks {2a, 2a+1}, a=0..6,
        plus one leftover 1-column block {14}.
      - Each 2-column block is a 2x2 x 2x2 tensor, decomposed by Strassen (rank 7).
        Because the V-support (columns) and W-support (rows k*n+i) of different
        blocks are disjoint, the ranks add: 7 * 7 = 49.
      - The leftover single column is a 2x2 x 2x1 tensor of rank 4 (schoolbook).
      - Total rank = 49 + 4 = 53.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()

    # Local Strassen factors (local k-index 0,1)
    sU, sV, sW = _build_strassen_2x2x2()
    srank = sU.shape[1]  # 7

    # Block layout: 7 two-column blocks (cols 0..13) + 1 one-column block (col 14)
    n_blocks = p // 2          # 7
    leftover = p - 2 * n_blocks  # 1
    rank = n_blocks * srank + n * m * leftover  # 49 + 4 = 53

    dim_U = n * m     # 4
    dim_V = m * p     # 30
    dim_W = p * n     # 30

    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))

    col = 0
    # --- 2-column blocks using Strassen ---
    for a in range(n_blocks):
        base_k = 2 * a  # global columns are base_k (local 0) and base_k+1 (local 1)
        for r in range(srank):
            U[:, col] = sU[:, r]
            for jj in range(m):
                for kk in range(2):
                    V[jj * p + (base_k + kk), col] = sV[jj * 2 + kk, r]
            for ii in range(n):
                for kk in range(2):
                    W[(base_k + kk) * n + ii, col] = sW[kk * 2 + ii, r]
            col += 1

    # --- leftover single column using schoolbook (rank 4) ---
    if leftover:
        base_k = 2 * n_blocks
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + base_k, col] = 1
                W[base_k * n + i, col] = 1
                col += 1

    # Verify; fall back to schoolbook if anything is off or if time is tight.
    if col != rank:
        U, V, W = _build_schoolbook(n, m, p)
    elif time.time() - t0 < _DEADLINE_SECONDS * 0.8:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        U, V, W = _build_schoolbook(n, m, p)
    else:
        U, V, W = _build_schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
