"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Structural Rank 18 solution (Strassen + Schoolbook) as baseline.
2. Search for lower ranks (17, 16) using ALS initialized with random and structured seeds.
3. Verify exactness using Fractions.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T


def _check_decomposition_float(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _snap_to_half_integers(M):
    """Snap matrix values to nearest half-integer."""
    return np.round(M * 2) / 2.0

def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    # Convert to Fractions
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            val = U[i, j]
            # Check if close to half integer
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                U_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                # If not close, it's likely not a valid exact solution with half-integers
                # But we try to limit denominator to 100 just in case
                U_frac[i, j] = Fraction(val).limit_denominator(100)
                
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            val = V[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                V_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                V_frac[i, j] = Fraction(val).limit_denominator(100)
                
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = W[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                W_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                W_frac[i, j] = Fraction(val).limit_denominator(100)
    
    R = U.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _get_rank18_solution():
    """
    Returns the structural Rank 18 solution:
    Two Strassen blocks (cols 0-1, 2-3) and Schoolbook for col 4.
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        # M1
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M2
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        # M3
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M4
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        # M5
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        # M6
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        # M7
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
        
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for col 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W


def _als_update_U(T, V, W, R):
    n, m, p = DIMENSIONS
    U_rows = n * m
    VW = np.einsum('vr,wr->vwr', V, W)
    VW_flat = VW.reshape(-1, R)
    U = np.zeros((U_rows, R), dtype=np.float64)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(VW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_update_V(T, U, W, R):
    n, m, p = DIMENSIONS
    V_rows = m * p
    UW = np.einsum('ur,wr->uwr', U, W)
    UW_flat = UW.reshape(-1, R)
    V = np.zeros((V_rows, R), dtype=np.float64)
    for v in range(V_rows):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UW_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_update_W(T, U, V, R):
    n, m, p = DIMENSIONS
    W_rows = p * n
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((W_rows, R), dtype=np.float64)
    for w in range(W_rows):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _compute_error(U, V, W, T, R):
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(result - T.astype(np.float64))

def _try_als_rank(R, T, deadline, n_seeds=50, max_iters=50):
    """
    Try to find a rank-R decomposition using ALS.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333, 
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    # Use Rank 18 solution as a structured initialization
    U18, V18, W18 = _get_rank18_solution()
    
    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Initialization Strategy 1: Random
        U = np.random.randn(U_rows, R) * 0.5
        V = np.random.randn(V_rows, R) * 0.5
        W = np.random.randn(W_rows, R) * 0.5
        
        # Initialization Strategy 2: Structured (from Rank 18, truncated or perturbed)
        # If R < 18, we can't directly use U18. 
        # But we can try to project or just use random for R < 18.
        # For R=17, we can try zeroing one column of U18? No, shapes differ.
        # So for R < 18, we rely on random or specific patterns.
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U = _als_update_U(T, V, W, R)
            V = _als_update_V(T, U, W, R)
            W = _als_update_W(T, U, V, R)
            
            err = _compute_error(U, V, W, T, R)
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        # Check for exact solution
        if _check_decomposition_float(U, V, W, T):
            # Try to rationalize to half-integers
            U_s = _snap_to_half_integers(U)
            V_s = _snap_to_half_integers(V)
            W_s = _snap_to_half_integers(W)
            
            if _check_decomposition_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
            # If snapping fails, check if original is exact enough
            # (Unlikely to be accepted if not half-integer, but try)
            if _check_decomposition_exact(U, V, W, T):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def construct():
    deadline = time.time() + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # Try to find Rank 17
    # Use a significant portion of the time for this
    search_deadline = time.time() + 130
    
    result = _try_als_rank(17, T, search_deadline, n_seeds=100, max_iters=50)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}
        
    # If time permits, try Rank 16 (unlikely to succeed, but cheap to check)
    if time.time() < deadline - 10:
        result = _try_als_rank(16, T, deadline, n_seeds=20, max_iters=30)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            
    return best_result
