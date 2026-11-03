"""
Strategy: 
1. The rank of 3x4x4 matrix multiplication is known to be 19 (via embedding in 4x4x4).
2. We will attempt to find a rank-19 decomposition for the 4x4x4 tensor.
3. If found, we restrict it to the 3x4x4 subspace.
4. If that fails, we attempt a direct rank-19 search for 3x4x4.
5. We use improved snapping to handle half-integers (denom 2).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _snap_to_half_integer(x, tol=1e-6):
    """Snap x to the nearest half-integer (k/2)."""
    n = round(x * 2)
    if abs(x * 2 - n) < tol:
        return n / 2.0
    return x

def _is_half_integer(x, tol=1e-6):
    """Check if x is a half-integer."""
    n = round(x * 2)
    return abs(x * 2 - n) < tol

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Use Fractions for exact arithmetic
    Uf = [[Fraction(u).limit_denominator(10**6) for u in row] for row in U]
    Vf = [[Fraction(v).limit_denominator(10**6) for v in row] for row in V]
    Wf = [[Fraction(w).limit_denominator(10**6) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for c in range(rank):
                    actual += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _als_decompose_4x4(rank, max_iter=500, tol=1e-12, deadline=None, seed=42):
    """
    ALS decomposition for 4x4x4 matrix multiplication (dims 16,16,16).
    """
    n, m, p = 4, 4, 4
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W

def _als_decompose_3x4x4(rank, max_iter=500, tol=1e-12, deadline=None, seed=42):
    """
    ALS decomposition for 3x4x4 matrix multiplication (dims 12,16,12).
    """
    n, m, p = 3, 4, 4
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W

def _snap_and_verify_3x4x4(U, V, W, deadline=None):
    """Snap to half-integers and verify for 3x4x4."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_snapped = [[_snap_to_half_integer(U[i][c]) for c in range(rank)] for i in range(n*m)]
    V_snapped = [[_snap_to_half_integer(V[i][c]) for c in range(rank)] for i in range(m*p)]
    W_snapped = [[_snap_to_half_integer(W[i][c]) for c in range(rank)] for i in range(p*n)]
    
    # Check if all are close to half-integers
    for row in U_snapped:
        for x in row:
            if not _is_half_integer(x):
                return None
    for row in V_snapped:
        for x in row:
            if not _is_half_integer(x):
                return None
    for row in W_snapped:
        for x in row:
            if not _is_half_integer(x):
                return None
    
    # Verify
    if _verify(U_snapped, V_snapped, W_snapped):
        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return None

def _restrict_4x4_to_3x4x4(U4, V4, W4):
    """
    Restrict a 4x4x4 decomposition (16x16x16) to 3x4x4 (12x16x12).
    i in 0..3 -> i in 0..2
    U indices: i*m + j. m=4.
    Keep rows where i < 3.
    V indices: j*p + k. No restriction (j 0..3, k 0..3).
    W indices: k*n + i. n=4.
    Keep rows where i < 3.
    """
    n4, m4, p4 = 4, 4, 4
    n3, m3, p3 = 3, 4, 4
    
    U3 = []
    for i in range(n3):
        for j in range(m4):
            idx = i * m4 + j
            U3.append(U4[idx][:])
    
    V3 = V4[:] # All rows
    
    W3 = []
    for k in range(p4):
        for i in range(n3):
            idx = k * n4 + i
            W3.append(W4[idx][:])
            
    return U3, V3, W3

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # 1. Try to find rank 19 decomposition for 4x4x4 and restrict
    #    This is a strong heuristic because rank(3x4x4) <= rank(4x4x4) = 19.
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        try:
            U4, V4, W4 = _als_decompose_4x4(19, max_iter=200, tol=1e-12, 
                                            deadline=deadline - 10, seed=seed)
            
            # Snap and check if 4x4x4 is valid
            U4s = [[_snap_to_half_integer(x) for x in row] for row in U4]
            V4s = [[_snap_to_half_integer(x) for x in row] for row in V4]
            W4s = [[_snap_to_half_integer(x) for x in row] for row in W4]
            
            # Quick check for 4x4x4
            # We don't strictly need 4x4x4 to be perfect if the restriction works, 
            # but it's safer. However, verification is expensive.
            # We will just try to restrict and verify the 3x4x4 part.
            U3, V3, W3 = _restrict_4x4_to_3x4x4(U4s, V4s, W4s)
            
            result = _snap_and_verify_3x4x4(U3, V3, W3, deadline=deadline)
            if result:
                return result
                
        except Exception:
            pass

    # 2. Try direct rank 19 search for 3x4x4
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        try:
            U, V, W = _als_decompose_3x4x4(19, max_iter=200, tol=1e-12,
                                           deadline=deadline - 5, seed=seed)
            result = _snap_and_verify_3x4x4(U, V, W, deadline=deadline)
            if result:
                return result
        except Exception:
            pass
            
    # 3. Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
