"""Rewritten program: search for a rank-minimal exact tensor decomposition of 3x4 * 4x5 matmul."""
import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p):
    """Check that U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}
    i.e., sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if (i,j,k) matches the matmul identity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the matmul tensor T[i,j,k] = 1 for all i,j,k (since matmul C[i,k] = sum_j A[i,j]B[j,k])
    # Actually the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_fraction_matrix(M, tol=1e-9):
    """Convert a float matrix to Fraction matrix, snapping near half-integers."""
    from fractions import Fraction
    rows = []
    for row in M:
        frow = []
        for v in row:
            # Try to snap to nearest half-integer
            snapped = round(v * 2) / 2
            if abs(v - snapped) < tol:
                frow.append(Fraction(snapped).limit_denominator(1000000))
            else:
                frow.append(Fraction(v).limit_denominator(1000000))
        rows.append(frow)
    return rows

def _try_rank(n, m, p, target_rank, seed=0, time_budget=30):
    """Try to find a decomposition of rank target_rank using random initialization + Newton-like refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize U, V, W randomly
    U = rng.randn(n*m, target_rank) * 0.1
    V = rng.randn(m*p, target_rank) * 0.1
    W = rng.randn(p*n, target_rank) * 0.1
    
    # We want: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p equations in U,V,W (3*n*m*p*target_rank variables but structured)
    # Use gradient descent / alternating minimization
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(2000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute residuals
        residuals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residuals[(i,j,k)] = s - 1.0
        
        err = max(abs(v) for v in residuals.values())
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-10:
            break
        
        # Update each factor by least squares holding others fixed
        # Update U: for each (i,j), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
        # This is linear in U[i*m+j,:] if we define A[i*m+j,k,c] = V[j*p+k,c]*W[k*n+i,c]
        # Then sum_c A[i*m+j,k,c] * U[i*m+j,c] = 1 for all k
        # Solve least squares for each (i,j)
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c (V[j*p+k,c]*W[k*n+i,c]) * U[i*m+j,c] = 1
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for c in range(target_rank):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(range(target_rank) if False else range(target_rank)):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = sol
                except:
                    pass
    
    if best_err < 1e-8:
        # Try to snap to half-integers
        U_frac = _to_fraction_matrix(best_U)
        V_frac = _to_fraction_matrix(best_V)
        W_frac = _to_fraction_matrix(best_W)
        # Verify with fractions
        max_err_frac = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(target_rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    err = abs(float(s - 1))
                    if err > max_err_frac:
                        max_err_frac = err
        if max_err_frac < 1e-6:
            return U_frac, V_frac, W_frac, max_err_frac
        else:
            return None, None, None, best_err
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # must finish within 120s, use 110s for safety
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Schoolbook rank = n*m*p = 60
    # Try ranks from 20 upward
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, get the schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try to find better ranks
    for target_rank in range(20, min(40, best_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            U, V, W, err = _try_rank(n, m, p, target_rank, seed=seed, time_budget=15)
            if U is not None:
                best_result = (U, V, W, target_rank)
                found = True
                break
        if found:
            break
    
    # Return the best result found
    U, V, W, rank = best_result
    
    # Convert to lists if they are Fraction matrices
    if isinstance(U, list) and len(U) > 0 and isinstance(U[0], list) and isinstance(U[0][0], Fraction):
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        # But the task says integer/half-integer entries, so let's keep them as fractions converted to float
        # Actually, let's return them as lists of floats that are exactly half-integers
    else:
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out, "V": V_out, "W": W_out}
