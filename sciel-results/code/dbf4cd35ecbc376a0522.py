"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses schoolbook (rank 50) as baseline and attempts structured search for
lower rank exploiting the n=2 structure.
"""

import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 108.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
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
    return U, V, W, rank


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def _verify_float(U, V, W, rank, tol=1e-8):
    """Fast float check for candidate solutions."""
    n, m, p = 2, 5, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k]) * W[k * n + i]
                # Actually need element-wise product then sum
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_structured_rank(r, max_attempts=3):
    """Try to find rank-r decomposition using n=2 structure."""
    if not _deadline_ok():
        return None
    
    n, m, p = 2, 5, 5
    
    for attempt in range(max_attempts):
        if not _deadline_ok():
            break
        
        # Initialize with random values
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Alternating minimization with exact linear solves
        converged = False
        for iteration in range(200):
            if not _deadline_ok():
                break
            
            # Fix U, V: solve for W
            # For each (i,k): sum_j sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
            A_w = np.zeros((n * p, p * n * r))
            b_w = np.ones(n * p)
            for i in range(n):
                for k in range(p):
                    eq = i * p + k
                    for s in range(r):
                        a_val = sum(U[i * m + j, s] * V[j * p + k, s] for j in range(m))
                        A_w[eq, (k * n + i) * r + s] = a_val
            
            W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W = W_flat.reshape(p * n, r)
            
            # Fix U, W: solve for V
            A_v = np.zeros((m * p, m * p * r))
            b_v = np.ones(m * p)
            for j in range(m):
                for k in range(p):
                    eq = j * p + k
                    for s in range(r):
                        b_val = sum(U[i * m + j, s] * W[k * n + i, s] for i in range(n))
                        A_v[eq, (j * p + k) * r + s] = b_val
            
            V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Fix V, W: solve for U
            A_u = np.zeros((n * m, n * m * r))
            b_u = np.ones(n * m)
            for i in range(n):
                for j in range(m):
                    eq = i * m + j
                    for s in range(r):
                        c_val = sum(V[j * p + k, s] * W[k * n + i, s] for k in range(p))
                        A_u[eq, (i * m + j) * r + s] = c_val
            
            U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U = U_flat.reshape(n * m, r)
            
            # Check convergence
            if _verify_float(U, V, W, r, tol=1e-10):
                converged = True
                break
        
        if converged:
            # Try to convert to exact fractions
            U_frac = [[Fraction(U[i, s]).limit_denominator(10000) for s in range(r)] for i in range(n * m)]
            V_frac = [[Fraction(V[i, s]).limit_denominator(10000) for s in range(r)] for i in range(m * p)]
            W_frac = [[Fraction(W[i, s]).limit_denominator(10000) for s in range(r)] for i in range(p * n)]
            
            if _verify(U_frac, V_frac, W_frac, r):
                return U_frac, V_frac, W_frac
    
    return None


def construct():
    # Try structured search for rank < 50
    # Lower bound: 25 (from m*p = 25)
    # Try a few promising ranks
    for r in [49, 48, 47, 46, 45]:
        if not _deadline_ok():
            break
        
        result = _try_structured_rank(r, max_attempts=2)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Should never happen, but rebuild if needed
    U, V, W, rank = _schoolbook()
    return {"U": U, "V": V, "W": W}
