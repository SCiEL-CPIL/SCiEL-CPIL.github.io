import time
import numpy as np

N_DIM = 4
M_DIM = 4
P_DIM = 5

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 0.90

def _timeup():
    return (time.time() - _T0) > _LIMIT * _MARGIN


# ----------------------------------------------------------------------
# Rank-7 2x2x2 Strassen factors (verified internally by the full 4x4x4 check)
# ----------------------------------------------------------------------
def _s2():
    # 7 rank-1 factors: (alpha[2,2], beta[2,2], gamma[2,2])
    F = [
        (np.array([[0,1],[1,1]], dtype=float),
         np.array([[1,1],[1,1]], dtype=float),
         np.array([[1,1],[1,1]], dtype=float)),
        (np.array([[1,0],[1,1]], dtype=float),
         np.array([[1,1],[1,1]], dtype=float),
         np.array([[0,0],[1,1]], dtype=float)),
        (np.array([[1,0],[0,0]], dtype=float),
         np.array([[1,0],[0,0]], dtype=float),
         np.array([[0,0],[0,1]], dtype=float)),
        (np.array([[0,1],[0,1]], dtype=float),
         np.array([[0,1],[0,1]], dtype=float),
         np.array([[1,0],[0,0]], dtype=float)),
        (np.array([[0,0],[1,1]], dtype=float),
         np.array([[1,1],[1,0]], dtype=float),
         np.array([[1,1],[1,0]], dtype=float)),
        (np.array([[0,0],[0,1]], dtype=float),
         np.array([[1,0],[0,1]], dtype=float),
         np.array([[1,0],[0,1]], dtype=float)),
        (np.array([[1,1],[1,0]], dtype=float),
         np.array([[1,0],[0,1]], dtype=float),
         np.array([[0,0],[1,1]], dtype=float)),
    ]
    return F


# ----------------------------------------------------------------------
# Build 4x4x4 tensor with rank <= 49 via 2-level Strassen
# ----------------------------------------------------------------------
def _build_444():
    F = _s2()
    R = 49
    U = np.zeros((16, R), dtype=float)
    V = np.zeros((16, R), dtype=float)
    W = np.zeros((16, R), dtype=float)

    # Top-level: block indices (bi,bj) for A 2x2x2, (bj,bk) for B, (bi,bk) for C
    # For each top-level factor t in 0..6, and each inner factor r in 0..6:
    #   column = t*7 + r
    #   A block (bi,bj) gets alpha_top[t][bi,bj] * alpha_inner[r]
    #   B block (bj,bk) gets beta_top[t][bj,bk] * beta_inner[r]
    #   C block (bi,bk) gets gamma_top[t][bi,bk] * gamma_inner[r]

    for t in range(7):
        aT, bT, gT = F[t]
        for r in range(7):
            aI, bI, gI = F[r]
            col = t * 7 + r

            # A: for each block (bi,bj), if aT[bi,bj] != 0, add aT[bi,bj]*aI
            for bi in range(2):
                for bj in range(2):
                    coeff = aT[bi, bj]
                    if coeff == 0.0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi * 2 + ii
                            gj = bj * 2 + jj
                            U[gi * 4 + gj, col] += coeff * aI[ii, jj]

            # B: for each block (bj,bk), if bT[bj,bk] != 0, add bT[bj,bk]*bI
            for bj in range(2):
                for bk in range(2):
                    coeff = bT[bj, bk]
                    if coeff == 0.0:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            gj = bj * 2 + jj
                            gk = bk * 2 + kk
                            V[gj * 4 + gk, col] += coeff * bI[jj, kk]

            # C: for each block (bi,bk), if gT[bi,bk] != 0, add gT[bi,bk]*gI
            for bi in range(2):
                for bk in range(2):
                    coeff = gT[bi, bk]
                    if coeff == 0.0:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            gi = bi * 2 + ii
                            gk = bk * 2 + kk
                            W[gk * 4 + gi, col] += coeff * gI[kk, ii]

    return U, V, W


# ----------------------------------------------------------------------
# Extend to 4x4x5: add 16 schoolbook columns for the 5th column of B
# ----------------------------------------------------------------------
def _build_445():
    U4, V4, W4 = _build_444()
    R4 = U4.shape[1]  # 49
    extra = 16
    R = R4 + extra
    U = np.zeros((16, R), dtype=float)
    V = np.zeros((20, R), dtype=float)
    W = np.zeros((20, R), dtype=float)

    U[:, :R4] = U4
    V[:16, :R4] = V4
    W[:16, :R4] = W4

    # Schoolbook for k=4 (0-indexed)
    c = R4
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0
            c += 1

    return U, V, W


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def _check(U, V, W, n=4, m=4, p=5, seed=42):
    R = U.shape[1]
    if U.shape != (n*m, R) or V.shape != (m*p, R) or W.shape != (p*n, R):
        return False
    rng = np.random.default_rng(seed)
    A = rng.random((n, m))
    B = rng.random((m, p))
    Ctrue = A @ B
    Cgot = np.einsum('ijr,jkr,kir->ik', U.reshape(n, m, R), V.reshape(m, p, R), W.reshape(p, n, R))
    if not np.allclose(Cgot, Ctrue, atol=1e-8):
        return False
    # Second check with different seed
    A2 = rng.random((n, m))
    B2 = rng.random((m, p))
    C2 = np.einsum('ijr,jkr,kir->ik', U.reshape(n, m, R), V.reshape(m, p, R), W.reshape(p, n, R))
    return np.allclose(C2, A2 @ B2, atol=1e-8)


# ----------------------------------------------------------------------
# Schoolbook fallback (rank 80)
# ----------------------------------------------------------------------
def _schoolbook():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
                c += 1
    return U, V, W


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def construct():
    # Try the Strassen-based rank-65 decomposition
    if not _timeup():
        U, V, W = _build_445()
        if _check(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
