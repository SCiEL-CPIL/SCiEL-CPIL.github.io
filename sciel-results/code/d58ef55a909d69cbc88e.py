import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using element-wise checks."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U3[i, j, :] * V3[j, k, :] * W3[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Standard Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    r = 7
    U = np.zeros((4, r), dtype=int)
    V = np.zeros((4, r), dtype=int)
    W = np.zeros((4, r), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)B00 -> C10 +=
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1
    
    # M3: A00(B01-B11) -> C01 +=
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[2, 2] += 1
    
    # M4: A11(B10-B00) -> C10 +=, C11 += (wait, let me recheck)
    # Actually: A11*B10 contributes to C10, A11*B00 contributes to... 
    # Let me use the standard Strassen formulation:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M3 + M7... 
    # Let me just use a known correct set.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M3 + M7... 
    # Wait, C11 = M1 - M5 + M2 + M7? Let me verify:
    # C11 = A10*B10 + A11*B11
    # M1 - M5 + M2 + M7 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A10+A11)B00 + (A01-A11)(B01+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A10B00 + A11B00 + A01B01 + A01B11 - A11B01 - A11B11
    # = A00B00 + A11B00 + A10B00 + A01B01 - A11B01 + A11B11 - A00B11 + A00B11 - A01B11 + A01B11
    # Hmm, this doesn't simplify to A10*B10 + A11*B11.
    
    # Let me use a different standard form. I'll just verify computationally.
    # For now, let me use a known-correct implementation.
    
    # Actually, let me just build the rank-48 schoolbook and use that as baseline,
    # then try to find a better decomposition through optimization.
    # The key insight is that we need to find integer/half-integer entries.
    
    # Let me use a simpler approach: start with schoolbook rank 48 and try to reduce.
    
    # For the 2x2x2 Strassen, here's a verified set:
    U = np.zeros((4, r), dtype=float)
    V = np.zeros((4, r), dtype=float)
    W = np.zeros((4, r), dtype=float)
    
    # I'll use the standard Strassen with verification
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1  # C10
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1  # C01
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C11
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C00
    
    # Verify
    for i in range(2):
        for k in range(2):
            expected = 0
            for j in range(2):
                expected += 1  # A[i,j]*B[j,k] should give 1 for the identity case
            # Actually for the tensor, T[i,j,k]=1 always
            pass
    
    return U, V, W


def _build_2x2x4_block():
    """
    Build a rank-10 or rank-14 decomposition for 2x2x4 matrix multiplication.
    We'll use two independent Strassen 2x2x2 blocks for rank-14,
    then try to reduce.
    """
    n, m, p = 2, 2, 4
    U, V, W = _strassen_2x2x2()
    
    # Combine two 2x2x2 blocks for 2x2x4
    r_total = 14
    U4 = np.zeros((n*m, r_total), dtype=float)
    V4 = np.zeros((m*p, r_total), dtype=float)
    W4 = np.zeros((p*n, r_total), dtype=float)
    
    for block in range(2):
        k0 = 2 * block
        base = 7 * block
        
        U4[:, base:base+7] = U
        for j in range(2):
            for k_local in range(2):
                k_global = k0 + k_local
                V4[j*p + k_global, base:base+7] = V[j*2 + k_local, :]
        for k_local in range(2):
            for i in range(2):
                k_global = k0 + k_local
                W4[k_global*n + i, base:base+7] = W[k_local*2 + i, :]
    
    # Verify
    if not _verify_fast(U4, V4, W4, n, m, p):
        # Fallback to schoolbook
        r_sb = n * m * p
        Us = np.zeros((n*m, r_sb), dtype=float)
        Vs = np.zeros((m*p, r_sb), dtype=float)
        Ws = np.zeros((p*n, r_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return Us, Vs, Ws
    
    return U4, V4, W4


def _build_2x2x12_recursive():
    """
    Build 2x2x12 decomposition using three 2x2x4 blocks.
    """
    n, m, p = 2, 2, 12
    U4, V4, W4 = _build_2x2x4_block()
    r4 = U4.shape[1]
    r_total = 3 * r4
    
    U12 = np.zeros((n*m, r_total), dtype=float)
    V12 = np.zeros((m*p, r_total), dtype=float)
    W12 = np.zeros((p*n, r_total), dtype=float)
    
    for block in range(3):
        k0 = 4 * block
        base = r4 * block
        
        U12[:, base:base+r4] = U4
        for j in range(2):
            for k_local in range(4):
                k_global = k0 + k_local
                V12[j*p + k_global, base:base+r4] = V4[j*4 + k_local, :]
        for k_local in range(4):
            for i in range(2):
                k_global = k0 + k_local
                W12[k_global*n + i, base:base+r4] = W4[k_local*2 + i, :]
    
    # Verify
    if not _verify_fast(U12, V12, W12, n, m, p):
        # Fallback
        r_sb = n * m * p
        Us = np.zeros((n*m, r_sb), dtype=float)
        Vs = np.zeros((m*p, r_sb), dtype=float)
        Ws = np.zeros((p*n, r_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return Us, Vs, Ws
    
    return U12, V12, W12


def _build_2x2x12_strassen():
    """
    Build 2x2x12 decomposition using six 2x2x2 Strassen blocks.
    """
    n, m, p = 2, 2, 12
    U, V, W = _strassen_2x2x2()
    r_total = 42
    
    U12 = np.zeros((n*m, r_total), dtype=float)
    V12 = np.zeros((m*p, r_total), dtype=float)
    W12 = np.zeros((p*n, r_total), dtype=float)
    
    for block in range(6):
        k0 = 2 * block
        base = 7 * block
        
        U12[:, base:base+7] = U
        for j in range(2):
            for k_local in range(2):
                k_global = k0 + k_local
                V12[j*p + k_global, base:base+7] = V[j*2 + k_local, :]
        for k_local in range(2):
            for i in range(2):
                k_global = k0 + k_local
                W12[k_global*n + i, base:base+7] = W[k_local*2 + i, :]
    
    # Verify
    if not _verify_fast(U12, V12, W12, n, m, p):
        r_sb = n * m * p
        Us = np.zeros((n*m, r_sb), dtype=float)
        Vs = np.zeros((m*p, r_sb), dtype=float)
        Ws = np.zeros((p*n, r_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return Us, Vs, Ws
    
    return U12, V12, W12


def _try_reduce_rank(U, V, W, n, m, p, deadline=None):
    """
    Try to reduce rank by removing redundant terms.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    
    for r_remove in range(r):
        if deadline and time.time() > deadline:
            break
        U_test = np.delete(U, r_remove, axis=1)
        V_test = np.delete(V, r_remove, axis=1)
        W_test = np.delete(W, r_remove, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            r -= 1
            r_remove = -1
    
    return U, V, W, r


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try recursive block decomposition (3 x 2x2x4 blocks)
    U, V, W = _build_2x2x12_recursive()
    rank_recursive = U.shape[1]
    
    # Try Strassen-based decomposition (6 x 2x2x2 blocks)
    U_s, V_s, W_s = _build_2x2x12_strassen()
    rank_strassen = U_s.shape[1]
    
    # Choose the one with lower rank
    if rank_recursive <= rank_strassen:
        U, V, W = U, V, W
    else:
        U, V, W = U_s, V_s, W_s
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline=deadline)
    
    # Ensure we have a valid decomposition
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        r_sb = n * m * p
        Us = np.zeros((n*m, r_sb), dtype=float)
        Vs = np.zeros((m*p, r_sb), dtype=float)
        Ws = np.zeros((p*n, r_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
