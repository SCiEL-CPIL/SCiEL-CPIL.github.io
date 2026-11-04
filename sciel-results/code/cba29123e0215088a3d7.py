"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # Safety margin before 150s kill
    
    # Try structured search for lower-rank decomposition
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_solution(n, m, p)

def _schoolbook_solution(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using numpy. Returns True if all tensor entries are within tol of 1.
    U, V, W are numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    rank = U.shape[1]
    # Precompute products for efficiency
    # For each c, we have U[:,c], V[:,c], W[:,c]
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            u_row = U[u_idx, :]
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                v_row = V[v_idx, :]
                w_row = W[w_idx, :]
                val = np.dot(u_row * v_row, w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """
    Exact verification using Fraction arithmetic.
    U, V, W are lists of lists of Fractions or integers.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True

def _structured_search(n, m, p, deadline):
    """
    Structured search for low-rank decomposition of 4x4x9 matrix multiplication.
    
    Strategy:
    1. Try block-structured approaches leveraging 4x4x4 substructure
    2. Use iterative refinement with good initializations
    3. Attempt exact rational reconstruction
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p  # schoolbook rank = 144
    
    # Try various rank targets
    # Lower bound considerations:
    # - Trivial: max(n*m, m*p, p*n) = 36
    # - For 4x4x4, Laderman gives rank 7
    # - For 4x4x9, we might expect rank around 20-40
    rank_targets = [20, 22, 24, 25, 26, 27, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    
    for target_rank in rank_targets:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple seeds
        num_attempts = 5
        for seed in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            sol = _attempt_decomposition(n, m, p, target_rank, seed, deadline)
            if sol is not None:
                best_solution = sol
                best_rank = target_rank
                break  # Move to next (lower) rank target
        
        if time.time() > deadline - 10:
            break
    
    return best_solution

def _attempt_decomposition(n, m, p, target_rank, seed, deadline):
    """
    Attempt to find a rank-target_rank decomposition.
    Uses iterative refinement with random initialization.
    """
    np.random.seed(seed * 1000 + 42)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.3
    V = np.random.randn(m * p, target_rank) * 0.3
    W = np.random.randn(p * n, target_rank) * 0.3
    
    # Iterative refinement - alternating least squares
    max_iterations = 80
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j,:]
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c] for k in range(p)
                A = np.zeros((p, target_rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k,:]
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # A[i, c] = U[i*m+j, c] * W[k*n+i, c] for i in range(n)
                A = np.zeros((n, target_rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i,:]
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in range(m)
                A = np.zeros((m, target_rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                u_row = U[u_idx, :]
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(u_row * V[v_idx, :], W[w_idx, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_err < 1e-8:
            # Try to reconstruct exact rational solution
            result = _reconstruct_exact(best_U, best_V, best_W, n, m, p, deadline)
            if result is not None:
                return result
            # Continue iterating to see if we can get better
            break
    
    # If we got close but not exact, try reconstruction
    if best_err < 1e-6:
        result = _reconstruct_exact(best_U, best_V, best_W, n, m, p, deadline)
        if result is not None:
            return result
    
    return None

def _reconstruct_exact(U, V, W, n, m, p, deadline):
    """
    Try to reconstruct an exact rational solution from floating point values.
    """
    if time.time() > deadline - 3:
        return None
    
    rank = U.shape[1]
    
    # Try different denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if time.time() > deadline - 2:
            break
        
        U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(rank)] for i in range(n * m)]
        V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(rank)] for i in range(m * p)]
        W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(rank)] for i in range(p * n)]
        
        if _verify_decomposition_exact(U_frac, V_frac, W_frac, n, m, p):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

# EVOLVE-BLOCK-END
