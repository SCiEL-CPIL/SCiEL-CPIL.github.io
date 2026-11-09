import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

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

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _als_decompose(n, m, p, rank, max_iter=500, tol=1e-10, seed=None):
    """ALS decomposition of the matrix multiplication tensor."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    for iteration in range(max_iter):
        # Update U
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    continue
                U_new[i*m+j, :] = x
        U = U_new
        
        # Update V
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    continue
                V_new[j*p+k, :] = x
        V = V_new
        
        # Update W
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    continue
                W_new[k*n+i, :] = x
        W = W_new
        
        # Check convergence
        if iteration % 10 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                break
    
    # Check if solution is valid
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-8:
        return U, V, W, max_err
    return None, None, None, max_err

def _round_and_verify(U, V, W, n, m, p):
    """Round to half-integers and verify exactness."""
    rank = len(U[0])
    
    # Round to nearest half-integer
    U_rounded = [[Fraction(int(round(2 * U[i][c])), 2) for c in range(rank)] for i in range(n*m)]
    V_rounded = [[Fraction(int(round(2 * V[i][c])), 2) for c in range(rank)] for i in range(m*p)]
    W_rounded = [[Fraction(int(round(2 * W[i][c])), 2) for c in range(rank)] for i in range(p*n)]
    
    # Verify exact decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_rounded[i*m+j][c] * V_rounded[j*p+k][c] * W_rounded[k*n+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_rounded, V_rounded, W_rounded
    return None, None, None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Known lower bound
    min_rank = max(n*m, m*p, p*n)
    
    # Try ranks from min_rank up to best_rank - 1
    # We prioritize lower ranks for higher score
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            seed = rank * 1000 + attempt
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=200, tol=1e-12, seed=seed)
            
            if U is not None:
                U_rounded, V_rounded, W_rounded = _round_and_verify(U, V, W, n, m, p)
                if U_rounded is not None:
                    best_rank = rank
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    # If we found a solution at the minimum possible rank, we're done
                    if rank == min_rank:
                        return best_result
                    break  # Move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
