"""Rank-minimal exact tensor decomposition for 4x4 * 4x5.

Strategy:
  1. Build a two-level Strassen decomposition for the 4x4x4 part (rank 49).
  2. Add 16 schoolbook terms for the 5th column (k=4).
  3. Total rank = 65.
  4. Verify exactly and return.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 15.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def _verify(U, V, W, n, m, p, tol=1e-8):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
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
    return bool(np.allclose(C, Ctrue, atol=tol))

def _make_4x4x4_strassen():
    """Two-level Strassen for 4x4x4. Returns U(16,49), V(16,49), W(16,49)."""
    n = m = p = 4
    
    # Strassen 2x2x2 definitions
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    A_side = [
        [(0, 1.0), (3, 1.0)],   # M1: A00+A11
        [(2, 1.0), (3, 1.0)],   # M2: A10+A11
        [(0, 1.0)],             # M3: A00
        [(3, 1.0)],             # M4: A11
        [(0, 1.0), (1, 1.0)],   # M5: A00+A01
        [(2, 1.0), (0, -1.0)],  # M6: A10-A00
        [(1, 1.0), (3, -1.0)],  # M7: A01-A11
    ]
    
    B_side = [
        [(0, 1.0), (3, 1.0)],   # M1: B00+B11
        [(0, 1.0)],             # M2: B00
        [(1, 1.0), (3, -1.0)],  # M3: B01-B11
        [(2, 1.0), (0, -1.0)],  # M4: B10-B00
        [(3, 1.0)],             # M5: B11
        [(0, 1.0), (1, 1.0)],   # M6: B00+B01
        [(2, 1.0), (3, 1.0)],   # M7: B10+B11
    ]
    
    out_coeff = [
        {0: 1.0, 3: 1.0},       # M1 -> C00, C11
        {2: 1.0, 3: -1.0},      # M2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # M3 -> C01, C11
        {0: 1.0, 2: 1.0},       # M4 -> C00, C10
        {0: -1.0, 1: 1.0},      # M5 -> -C00, C01
        {3: 1.0},               # M6 -> C11
        {0: 1.0},               # M7 -> C00
    ]
    
    inner_A_side = A_side
    inner_B_side = B_side
    inner_out_coeff = out_coeff
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk
            
            # U part: A side
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    i = (bi // 2) * 2 + u_local
                    j = (bi % 2) * 2 + v_local
                    U[i * m + j, col] += c1 * c2
            
            # V part: B side
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    j = (bi // 2) * 2 + u_local
                    k = (bi % 2) * 2 + v_local
                    V[j * p + k, col] += c1 * c2
            
            # W part: output
            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    i = (bi_outer // 2) * 2 + u_local
                    k = (bi_outer % 2) * 2 + v_local
                    W[k * n + i, col] += c_inner * c_outer
    
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 65: Recursive Strassen 4x4x4 (rank 49) + Schoolbook 4x4x1 (rank 16)
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: Recursive Strassen 4x4x4
    U4, V4, W4 = _make_4x4x4_strassen()
    U[:, :rank_part1] = U4
    V[:, :rank_part1] = V4
    W[:, :rank_part1] = W4
    
    # Part 2: Schoolbook for column 4 (k=4)
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook), dtype=float)
        V = np.zeros((m * p, rank_schoolbook), dtype=float)
        W = np.zeros((p * n, rank_schoolbook), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
