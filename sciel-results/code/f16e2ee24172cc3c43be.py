"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3x3 matmul."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    # T[i, j, l] = 1 for all valid (i in 0..n-1, j in 0..m-1, l in 0..p-1)
    # Because C[i,l] = sum_j A[i,j] B[j,l]; each term (i,j,l) appears with coefficient 1.
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i, j, l] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T = _build_tensor()
    R = len(U[0]) if isinstance(U, list) and U else U.shape[1]
    # Build the tensor from decomposition: T[i,j,l] = sum_c U[i*m+j, c]*V[j*p+l, c]*W[l*n+i, c]
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+l][c] * W[l*n+i][c]
                if abs(s - T[i, j, l]) > tol:
                    return False
    return True

def _try_direct_rank(R, max_iters=2000, seed=0):
    """Try to find a rank-R decomposition via random init + coordinate descent refinement.
    We parametrize and minimize ||T - decomposed||^2, then snap to rationals if close."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    best_err = float('inf')
    best = None
    for it in range(max_iters):
        # Random initialization with structure: mix of random and identity-like
        if it % 3 == 0:
            U = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(n*m)], dtype=float)
            V = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(p*n)], dtype=float)
        elif it % 3 == 1:
            # Initialize with some structure: first few columns are identity-like
            U = np.zeros((n*m, R), dtype=float)
            V = np.zeros((m*p, R), dtype=float)
            W = np.zeros((p*n, R), dtype=float)
            for c in range(min(R, n*m)):
                U[c, c] = 1.0
            for c in range(min(R, m*p)):
                V[c, c] = 1.0
            for c in range(min(R, p*n)):
                W[c, c] = 1.0
            # Add random perturbation
            U += rng.gauss(0, 0.1) * np.ones_like(U)
            V += rng.gauss(0, 0.1) * np.ones_like(V)
            W += rng.gauss(0, 0.1) * np.ones_like(W)
        else:
            # Sparse random init
            U = np.zeros((n*m, R), dtype=float)
            V = np.zeros((m*p, R), dtype=float)
            W = np.zeros((p*n, R), dtype=float)
            for c in range(R):
                idx_u = rng.randint(0, n*m-1)
                idx_v = rng.randint(0, m*p-1)
                idx_w = rng.randint(0, p*n-1)
                U[idx_u, c] = rng.choice([-1, 1])
                V[idx_v, c] = rng.choice([-1, 1])
                W[idx_w, c] = rng.choice([-1, 1])
        
        # Coordinate descent: update one factor at a time by least squares.
        cd_steps = 80 if R <= 14 else 60
        for _ in range(cd_steps):
            # Update U: for each i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    idx = i*m+j
                    M = np.zeros((p, R))
                    b = np.zeros(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l][c] * W[l*n+i][c]
                        b[l] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            # Update V (FIXED: l ranges over p, not m)
            for j in range(m):
                for l in range(p):
                    idx = j*p+l
                    M = np.zeros((n, R))
                    b = np.zeros(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j][c] * W[l*n+i][c]
                        b[i] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            # Update W
            for l in range(p):
                for i in range(n):
                    idx = l*n+i
                    M = np.zeros((m, R))
                    b = np.zeros(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j][c] * V[j*p+l][c]
                        b[j] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
        # Evaluate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i*m+j][c]*V[j*p+l][c]*W[l*n+i][c]
                    err += (s - 1.0)**2
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err < 1e-10:
            break
    if best is None:
        return None
    if best_err < 1e-6:
        U, V, W = best
        # High-precision polish: run additional coordinate descent with tighter convergence
        for _ in range(50):
            for i in range(n):
                for j in range(m):
                    idx = i*m+j
                    M = np.zeros((p, R))
                    b = np.zeros(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l][c] * W[l*n+i][c]
                        b[l] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            for j in range(m):
                for l in range(p):
                    idx = j*p+l
                    M = np.zeros((n, R))
                    b = np.zeros(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j][c] * W[l*n+i][c]
                        b[i] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            for l in range(p):
                for i in range(n):
                    idx = l*n+i
                    M = np.zeros((m, R))
                    b = np.zeros(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j][c] * V[j*p+l][c]
                        b[j] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
        
        # Convert to rationals with wider search
        U_r = [[_to_frac(x) for x in row] for row in U]
        V_r = [[_to_frac(x) for x in row] for row in V]
        W_r = [[_to_frac(x) for x in row] for row in W]
        # Verify exactly with Fraction
        if _verify_exact(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
        # Try with slightly different rounding
        U_r2 = [[_to_frac2(x) for x in row] for row in U]
        V_r2 = [[_to_frac2(x) for x in row] for row in V]
        W_r2 = [[_to_frac2(x) for x in row] for row in W]
        if _verify_exact(U_r2, V_r2, W_r2):
            return (U_r2, V_r2, W_r2)
    return None

def _to_frac(x):
    # Snap to nearest half-integer or integer, or a simple fraction
    r = round(x * 2) / 2.0
    if abs(x - r) < 1e-6:
        return r
    # Try fraction with small denominator
    for den in [1, 2, 3, 4, 5, 6]:
        f = round(x * den) / den
        if abs(x - f) < 1e-6:
            return f
    return x

def _to_frac2(x):
    # Alternative rounding: try more denominators
    for den in [1, 2, 3, 4, 5, 6, 8, 12]:
        f = round(x * den) / den
        if abs(x - f) < 1e-5:
            return f
    # Fall back to nearest 0.1
    f = round(x * 10) / 10.0
    if abs(x - f) < 1e-4:
        return f
    return x

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+l][c]) * Fraction(W[l*n+i][c])
                if s != 1:
                    return False
    return True

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
    deadline = time.time() + 115
    # Try ranks from 12 up to 18 (include 18 as it may be findable)
    for R in range(12, 19):
        if time.time() > deadline - 15:
            break
        # More seeds for lower ranks, fewer for higher
        max_seeds = 30 if R <= 14 else 15
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            max_iters = 200 if R <= 14 else 120
            res = _try_direct_rank(R, max_iters=max_iters, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
