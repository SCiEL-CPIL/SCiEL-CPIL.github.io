"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Hard-coded structured block decomposition (Strassen 2x2 blocks + schoolbook for col 4)
2. Aggressive rank-17 search using ALS with multiple initializations
3. Rational polish (half-integer / small-denominator snapping + exact verification)
4. Schoolbook fallback (rank 20)

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    """Reconstruct tensor from factor matrices."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            num = round(x * denom)
            candidate = Fraction(int(num), denom)
            if abs(x - float(candidate)) < 1e-9:
                return candidate
        f = Fraction(x).limit_denominator(128)
        return f

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Try to snap U,V,W to small-denominator rationals that still decompose exactly."""
    for denom in [1, 2, 4, 8, 16, 32, 64]:
        U_s = _snap(U, denom)
        V_s = _snap(V, denom)
        W_s = _snap(W, denom)
        if _check_decomposition(U_s, V_s, W_s, T, tol=1e-6):
            if _check_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    for dU in [1, 2, 4]:
        for dV in [1, 2, 4]:
            for dW in [1, 2, 4]:
                U_s = _snap(U, dU)
                V_s = _snap(V, dV)
                W_s = _snap(W, dW)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    for scale in [0.1, 0.05, 0.01]:
        for trial in range(10):
            U_p = U + np.random.randn(*U.shape) * scale
            V_p = V + np.random.randn(*V.shape) * scale
            W_p = W + np.random.randn(*W.shape) * scale
            for denom in [1, 2, 4]:
                U_s = _snap(U_p, denom)
                V_s = _snap(V_p, denom)
                W_s = _snap(W_p, denom)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None


def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _build_structured_decomposition():
    """
    Build a structured decomposition using Strassen for 2x2 column blocks
    and schoolbook for the remaining column.
    
    For 2x2 * 2x5:
    - Columns 0,1: Strassen (rank 7)
    - Columns 2,3: Strassen (rank 7)  
    - Column 4: schoolbook (rank 4)
    Total: rank 18
    """
    n, m, p = DIMENSIONS
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        # M1: f=A00+A11, g=B[0,col0]+B[1,col1]
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M2: f=A10+A11, g=B[0,col0]
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        # M3: f=A00, g=B[0,col1]-B[1,col1]
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M4: f=A11, g=B[1,col0]-B[0,col0]
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        # M5: f=A00+A01, g=B[1,col1]
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        # M6: f=A10-A00, g=B[0,col0]+B[0,col1]
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        # M7: f=A01-A11, g=B[1,col0]+B[1,col1]
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W


def _try_rank17_als(T, deadline):
    """
    Try to find a rank-17 decomposition using ALS with multiple initializations.
    Start from the rank-18 solution and try to merge/replace one term.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    # Try different strategies for initializing the rank-17 search
    strategies = []
    
    # Strategy 1: Remove the last schoolbook term and re-solve
    for seed in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize with rank-18 solution, zero out one column, add noise
        U = U_struct.copy()
        V = V_struct.copy()
        W = W_struct.copy()
        
        # Remove one column (try different ones)
        remove_col = seed % 18
        
        U_new = np.zeros((n * m, target_rank), dtype=np.float64)
        V_new = np.zeros((m * p, target_rank), dtype=np.float64)
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Copy all columns except the removed one
        idx = 0
        for i in range(18):
            if i != remove_col:
                U_new[:, idx] = U[:, i]
                V_new[:, idx] = V[:, i]
                W_new[:, idx] = W[:, i]
                idx += 1
        
        # Add small noise to the last column
        U_new[:, -1] += np.random.randn(n * m) * 0.1
        V_new[:, -1] += np.random.randn(m * p) * 0.1
        W_new[:, -1] += np.random.randn(p * n) * 0.1
        
        # Run ALS
        for it in range(100):
            if time.time() > deadline:
                break
            U_new = _als_update_U(T, V_new, W_new)
            V_new = _als_update_V(T, U_new, W_new)
            W_new = _als_update_W(T, U_new, V_new)
        
        if _check_decomposition(U_new, V_new, W_new, T, tol=1e-7):
            polished = _polish_rational(U_new, V_new, W_new, T)
            if polished is not None:
                return polished
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _try_rank17_structured(T, deadline):
    """
    Try a structured rank-17 decomposition by sharing one term between blocks.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    # Try sharing the M1 term (A00+A11 with B[0,col0]+B[1,col1])
    # between the two Strassen blocks by using a combined coefficient
    
    U = np.zeros((n * m, target_rank), dtype=np.float64)
    V = np.zeros((m * p, target_rank), dtype=np.float64)
    W = np.zeros((p * n, target_rank), dtype=np.float64)
    
    # Shared term: f=A00+A11, g=B[0,0]+B[1,1]+B[0,2]+B[1,3]
    # This gives contributions to both (0,1) and (2,3) blocks
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0*p+0, r] = 1; V[1*p+1, r] = 1
    V[0*p+2, r] = 1; V[1*p+3, r] = 1
    W[0*n+0, r] = 1; W[1*n+1, r] = 1
    W[2*n+0, r] = 1; W[3*n+1, r] = 1
    
    # Remaining 6 terms for block (0,1)
    # M2: f=A10+A11, g=B[0,0]
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0*p+0, r] = 1
    W[0*n+1, r] = 1; W[1*n+1, r] = -1
    
    # M3: f=A00, g=B[0,1]-B[1,1]
    r = 2
    U[0, r] = 1
    V[0*p+1, r] = 1; V[1*p+1, r] = -1
    W[1*n+0, r] = 1; W[1*n+1, r] = 1
    
    # M4: f=A11, g=B[1,0]-B[0,0]
    r = 3
    U[3, r] = 1
    V[1*p+0, r] = 1; V[0*p+0, r] = -1
    W[0*n+0, r] = 1; W[0*n+1, r] = 1
    
    # M5: f=A00+A01, g=B[1,1]
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[1*p+1, r] = 1
    W[0*n+0, r] = -1; W[1*n+0, r] = 1
    
    # M6: f=A10-A00, g=B[0,0]+B[0,1]
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0*p+0, r] = 1; V[0*p+1, r] = 1
    W[1*n+1, r] = 1
    
    # M7: f=A01-A11, g=B[1,0]+B[1,1]
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[1*p+0, r] = 1; V[1*p+1, r] = 1
    W[0*n+0, r] = 1
    
    # Remaining 6 terms for block (2,3)
    # M8: f=A10+A11, g=B[0,2]
    r = 7
    U[2, r] = 1; U[3, r] = 1
    V[0*p+2, r] = 1
    W[2*n+1, r] = 1; W[3*n+1, r] = -1
    
    # M9: f=A00, g=B[0,3]-B[1,3]
    r = 8
    U[0, r] = 1
    V[0*p+3, r] = 1; V[1*p+3, r] = -1
    W[3*n+0, r] = 1; W[3*n+1, r] = 1
    
    # M10: f=A11, g=B[1,2]-B[0,2]
    r = 9
    U[3, r] = 1
    V[1*p+2, r] = 1; V[0*p+2, r] = -1
    W[2*n+0, r] = 1; W[2*n+1, r] = 1
    
    # M11: f=A00+A01, g=B[1,3]
    r = 10
    U[0, r] = 1; U[1, r] = 1
    V[1*p+3, r] = 1
    W[2*n+0, r] = -1; W[3*n+0, r] = 1
    
    # M12: f=A10-A00, g=B[0,2]+B[0,3]
    r = 11
    U[2, r] = 1; U[0, r] = -1
    V[0*p+2, r] = 1; V[0*p+3, r] = 1
    W[3*n+1, r] = 1
    
    # M13: f=A01-A11, g=B[1,2]+B[1,3]
    r = 12
    U[1, r] = 1; U[3, r] = -1
    V[1*p+2, r] = 1; V[1*p+3, r] = 1
    W[2*n+0, r] = 1
    
    # 4 terms for column 4
    r = 13
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 14
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 16
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    # Check if this works
    if _check_decomposition(U, V, W, T, tol=1e-8):
        polished = _polish_rational(U, V, W, T)
        if polished is not None:
            return polished
        return U.tolist(), V.tolist(), W.tolist()
    
    # If not, try ALS refinement
    for it in range(50):
        if time.time() > deadline:
            break
        U = _als_update_U(T, V, W)
        V = _als_update_V(T, U, W)
        W = _als_update_W(T, U, V)
    
    if _check_decomposition(U, V, W, T, tol=1e-7):
        polished = _polish_rational(U, V, W, T)
        if polished is not None:
            return polished
        return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Build the structured decomposition (rank 18)
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    # Verify the structured decomposition is valid
    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Try structured rank-17 first (fast)
    if time.time() < deadline:
        result17 = _try_rank17_structured(T, deadline)
        if result17 is not None:
            U_l, V_l, W_l = result17
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}
    
    # Try ALS-based rank-17 search
    if time.time() < deadline:
        result17_als = _try_rank17_als(T, deadline)
        if result17_als is not None:
            U_l, V_l, W_l = result17_als
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}
    
    # If no reduction found, return the structured solution (rank 18)
    polished = _polish_rational(U_struct, V_struct, W_struct, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}
    
    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
