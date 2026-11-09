import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]
START = time.time()
DEADLINE_FRAC = 0.85

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    if not np.allclose(C, Ctrue, atol=tol):
        return False
    A2 = np.random.default_rng(7).random((n, m))
    B2 = np.random.default_rng(11).random((m, p))
    C2 = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C2, A2 @ B2, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2_blocks():
    S = [
        ([0, 1], [1, 1]),
        ([1, 0], [1, 1]),
        ([0, 0], [0, 1]),
        ([1, 1], [0, 0]),
        ([1, 0], [0, 0]),
        ([0, 1], [0, 1]),
        ([1, 1], [1, 0]),
    ]
    Cuse = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    return S, Cuse

def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    S, Cuse = _strassen_2x2_blocks()
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    for t, (ac, bc) in enumerate(S):
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = ac[i][j]
            for k in range(2):
                V[j * ncolsk + k, t] = bc[j][k]
    for i in range(2):
        for k in range(2):
            for t in Cuse[i * 2 + k]:
                W[k * nrows + i, t] = 1.0
    return U, V, W

def _build_445_strassen(n, m, p):
    """4x4x4 Strassen (rank 49) + 16 schoolbook for k=4, total rank 65."""
    S, Cuse = _strassen_2x2_blocks()
    R0 = 7
    U0, V0, W0 = _make_2x2x2_strassen(2, 2, 2)
    U = np.zeros((n * m, 7 * R0), dtype=float)
    V = np.zeros((m * p, 7 * R0), dtype=float)
    W = np.zeros((p * n, 7 * R0), dtype=float)
    for t in range(7):
        ac = S[t][0]; bc = S[t][1]
        for ri in range(2):
            for ci in range(2):
                acoef = ac[ri][ci]
                if acoef == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bcoef = bc[ci2][ki]
                        if bcoef == 0:
                            continue
                        for tp in range(R0):
                            col = t * R0 + tp
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] += acoef * U0[ii * 2 + jj, tp]
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, col] += bcoef * V0[jj2 * 2 + kk, tp]
                        for tp in Cuse[ri * 2 + ki]:
                            col = t * R0 + tp
                            for ii in range(2):
                                for kk in range(2):
                                    gi = ri * 2 + ii
                                    gk = ki * 2 + kk
                                    W[gk * n + gi, col] += W0[kk * 2 + ii, tp]
    # Append schoolbook for k=4
    R = U.shape[1]
    extra = n * m
    Ufull = np.zeros((n * m, R + extra), dtype=float)
    Vfull = np.zeros((m * p, R + extra), dtype=float)
    Wfull = np.zeros((p * n, R + extra), dtype=float)
    Ufull[:, :R] = U
    Vfull[:, :R] = V
    Wfull[:, :R] = W
    k = 4
    c = R
    for i in range(n):
        for j in range(m):
            Ufull[i * m + j, c] = 1.0
            Vfull[j * p + k, c] = 1.0
            Wfull[k * n + i, c] = 1.0
            c += 1
    return Ufull, Vfull, Wfull

def _solve_2x4x2_schoolbook():
    """Schoolbook for 2x4x2. Rank 16."""
    n, m, p = 2, 4, 2
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _build_445_rectangular(n, m, p):
    """
    Decompose 4x4x5 into four 2x4x2 blocks for cols 0-3 and one 4x4x1 block for col 4.
    Uses schoolbook for 2x4x2 blocks (rank 16 each).
    Total rank: 4 * 16 + 16 = 80.
    """
    # 4x4x4 part decomposed into 4 blocks of 2x4x2
    # Block 1: C[0:2, 0:2] from A[0:2, 0:4] and B[0:4, 0:2]
    # Block 2: C[0:2, 2:4] from A[0:2, 0:4] and B[0:4, 2:4]
    # Block 3: C[2:4, 0:2] from A[2:4, 0:4] and B[0:4, 0:2]
    # Block 4: C[2:4, 2:4] from A[2:4, 0:4] and B[0:4, 2:4]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # We need to map local indices (2x4x2) to global indices (4x4x4)
    # Local U: (2*4, R) -> Global U: (4*4, R)
    # Local V: (4*2, R) -> Global V: (4*4, R)
    # Local W: (2*2, R) -> Global W: (4*4, R)
    
    base_rank = 16
    total_rank = 4 * base_rank + (n * m) # 4x4x1 part
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Define the 4 blocks
    # (i_start, i_end, k_start, k_end)
    blocks = [
        (0, 2, 0, 2),
        (0, 2, 2, 4),
        (2, 4, 0, 2),
        (2, 4, 2, 4)
    ]
    
    for (i_s, i_e, k_s, k_e) in blocks:
        # Get schoolbook 2x4x2
        Ub, Vb, Wb = _solve_2x4x2_schoolbook()
        
        # Map to global
        # U global row: i * m + j. Local row: i_l * m_l + j.
        # i_l is 0 or 1. i = i_s + i_l.
        # j is 0..3.
        # V global row: j * p + k. Local row: j * p_l + k_l.
        # j is 0..3. k_l is 0 or 1. k = k_s + k_l.
        # W global row: k * n + i. Local row: k_l * n_l + i_l.
        # k_l is 0 or 1. k = k_s + k_l. i_l is 0 or 1. i = i_s + i_l.
        
        for r in range(base_rank):
            col = col_offset + r
            for i_l in range(2):
                for j in range(4):
                    U[(i_s + i_l) * m + j, col] = Ub[i_l * 4 + j, r]
            for j in range(4):
                for k_l in range(2):
                    V[j * p + (k_s + k_l), col] = Vb[j * 2 + k_l, r]
            for k_l in range(2):
                for i_l in range(2):
                    W[(k_s + k_l) * n + (i_s + i_l), col] = Wb[k_l * 2 + i_l, r]
        
        col_offset += base_rank
    
    # Add 4x4x1 block for k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_offset] = 1.0
            V[j * p + 4, col_offset] = 1.0
            W[4 * n + i, col_offset] = 1.0
            col_offset += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best = None

    # Candidate 1: Strassen composition (rank 65)
    if not _deadline():
        try:
            U2, V2, W2 = _build_445_strassen(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass

    # Candidate 2: Rectangular block decomposition (rank 80)
    if not _deadline():
        try:
            U3, V3, W3 = _build_445_rectangular(n, m, p)
            if _verify(U3, V3, W3, n, m, p):
                if best is None or U3.shape[1] < best[3]:
                    best = (U3, V3, W3, U3.shape[1])
        except Exception:
            pass

    # Candidate 3: schoolbook (rank 80)
    try:
        Us, Vs, Ws = _schoolbook(n, m, p)
        if _verify(Us, Vs, Ws, n, m, p):
            if best is None or Us.shape[1] < best[3]:
                best = (Us, Vs, Ws, Us.shape[1])
    except Exception:
        pass

    if best is None:
        Us, Vs, Ws = _schoolbook(n, m, p)
        best = (Us, Vs, Ws, Us.shape[1])

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
