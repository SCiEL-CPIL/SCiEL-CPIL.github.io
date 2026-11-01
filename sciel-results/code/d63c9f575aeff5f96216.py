"""
Matrix multiplication 2x3x3 - rank minimization
Target: rank 11 (known optimal for 2x3x3 matrix multiplication)
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s


def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = n, m, p
    rank = U.shape[1]
    # Compute tensor T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # and check against the matmul tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    # Check that all other entries are zero
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Determine if this is a "1" position
                i = u // m
                j = u % m
                jj = v // p
                k = v % p
                kk = w // n
                ii = w % n
                is_target = (i == ii) and (j == jj) and (k == kk)
                val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                if is_target:
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True


def _snap_to_half_integer(x, tol=1e-6):
    """Snap x to nearest half-integer if within tolerance."""
    h = round(2 * x) / 2.0
    if abs(x - h) < tol:
        return h
    return x


def _try_snap_and_verify(U, V, W, n, m, p):
    """Try to snap all entries to half-integers and verify."""
    U_s = [[_snap_to_half_integer(U[i][r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_s = [[_snap_to_half_integer(V[i][r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_s = [[_snap_to_half_integer(W[i][r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Check if all are half-integers
    for mat in [U_s, V_s, W_s]:
        for row in mat:
            for x in row:
                if abs(2 * x - round(2 * x)) > 1e-9:
                    return None
    
    if _check_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-8):
        return {"U": U_s, "V": V_s, "W": W_s}
    return None


def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _random_init(rank, n, m, p):
    """Random initialization for optimization."""
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    return U, V, W


def _residual(U, V, W, n, m, p):
    """Compute sum of squared residuals over all tensor entries."""
    rank = U.shape[1]
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u] * V[v], W[w])  # sum_r U[u,r]*V[v,r]*W[w,r]
                res += (val - 1.0) ** 2
    # Also need zero elsewhere - but this is 324 entries, expensive
    # For speed, only check the 18 target entries during optimization
    return res


def _full_residual(U, V, W, n, m, p):
    """Full residual including zero entries."""
    rank = U.shape[1]
    res = 0.0
    for u in range(n * m):
        i = u // m
        j = u % m
        for v in range(m * p):
            jj = v // p
            k = v % p
            for w in range(p * n):
                kk = w // n
                ii = w % n
                is_target = (i == ii) and (j == jj) and (k == kk)
                val = np.dot(U[u] * V[v], W[w])
                if is_target:
                    res += (val - 1.0) ** 2
                else:
                    res += val ** 2
    return res


def _optimize_from_init(U0, V0, W0, n, m, p, max_iter=200):
    """Use coordinate descent / simple gradient-free approach."""
    rank = U0.shape[1]
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    # Use scipy least_squares if available
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return U, V, W, _full_residual(U, V, W, n, m, p)
    
    def make_residual_func(fixed_U, fixed_V):
        def res(x):
            Wx = x.reshape(p * n, rank)
            r = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        val = np.dot(fixed_U[u] * fixed_V[v], Wx[w])
                        r += (val - 1.0) ** 2
            return np.sqrt(r)
        return res
    
    # Optimize W with U,V fixed
    x0 = W.ravel()
    res_w = make_residual_func(U, V)
    sol = least_squares(res_w, x0, max_nfev=1000)
    W = sol.x.reshape(p * n, rank)
    
    # Optimize V with U,W fixed
    def res_v(x):
        Vx = x.reshape(m * p, rank)
        r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(U[u] * Vx[v], W[w])
                    r += (val - 1.0) ** 2
        return np.sqrt(r)
    x0 = V.ravel()
    sol = least_squares(res_v, x0, max_nfev=1000)
    V = sol.x.reshape(m * p, rank)
    
    # Optimize U with V,W fixed
    def res_u(x):
        Ux = x.ravel() if hasattr(x, 'ravel') else x
        Ux = x.reshape(n * m, rank)
        r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(Ux[u] * V[v], W[w])
                    r += (val - 1.0) ** 2
        return np.sqrt(r)
    x0 = U.ravel()
    sol = least_squares(res_u, x0, max_nfev=1000)
    U = sol.x.reshape(n * m, rank)
    
    # Iterate a few times
    for _ in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        x0 = W.ravel()
        sol = least_squares(res_w, x0, max_nfev=500)
        W = sol.x.reshape(p * n, rank)
        
        sol = least_squares(res_v, V.ravel(), max_nfev=500)
        V = sol.x.reshape(m * p, rank)
        
        sol = least_squares(res_u, U.ravel(), max_nfev=500)
        U = sol.x.reshape(n * m, rank)
    
    return U, V, W, _full_residual(U, V, W, n, m, p)


def _try_rank(R, n, m, p, n_restarts=20):
    """Try to find a decomposition of rank R."""
    best_result = None
    best_res = float('inf')
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        U0, V0, W0 = _random_init(R, n, m, p)
        
        # Give some structure: initialize with partial schoolbook-like pattern
        # Set some entries to 1 to help convergence
        if restart % 3 == 0:
            # Random permutation of which (i,j,k) maps to which rank column
            perm = list(range(n * m * p))
            random.shuffle(perm)
            for col in range(min(R, n * m * p)):
                idx = perm[col] if col < len(perm) else col % (n * m * p)
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                if col < R - 1:
                    # Add some random perturbation
                    U0[i * m + j, col + 1] = 0.5
                    V0[j * p + k, col + 1] = 0.5
                    W0[k * n + i, col + 1] = 0.5
        
        U, V, W, res = _optimize_from_init(U0, V0, W0, n, m, p)
        
        if res < best_res:
            best_res = res
            # Try to snap
            snapped = _try_snap_and_verify(
                U.tolist(), V.tolist(), W.tolist(), n, m, p
            )
            if snapped is not None:
                return snapped
        
        if best_res < 1e-10:
            snapped = _try_snap_and_verify(
                U.tolist(), V.tolist(), W.tolist(), n, m, p
            )
            if snapped is not None:
                return snapped
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try rank 11 first (known optimal for 2x3x3)
    result = _try_rank(11, n, m, p, n_restarts=15)
    if result is not None:
        return result
    
    if time.time() - START_TIME > TIME_LIMIT - 20:
        return _schoolbook(n, m, p)
    
    # Try rank 12
    result = _try_rank(12, n, m, p, n_restarts=10)
    if result is not None:
        return result
    
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return _schoolbook(n, m, p)
    
    # Try rank 13
    result = _try_rank(13, n, m, p, n_restarts=5)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
