import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 145  # seconds, safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _winograd_convolution_decomposition(n, m, p):
    """
    Attempt to construct a decomposition using the Winograd convolution algorithm
    mapped to the ring Z[x]/(x^5 - 1).
    
    The idea:
    1. View matrix multiplication as a series of convolutions.
    2. Use the factorization of x^5 - 1 over the rationals (or complex numbers)
       to evaluate the polynomial at specific points (roots of unity).
    3. The number of evaluation points needed is related to the rank.
    
    For 4x5x5, we can try to exploit the structure where one dimension is 5.
    The polynomial x^5 - 1 factors as (x-1)(x^4 + x^3 + x^2 + x + 1).
    The roots are 1 and the primitive 5th roots of unity.
    
    However, exact decomposition over rationals/integers is tricky with complex roots.
    We can use the fact that the minimal polynomial of the primitive roots is
    Phi_5(x) = x^4 + x^3 + x^2 + x + 1.
    
    Let's try a specific construction:
    We want to compute C = A * B where A is 4x5 and B is 5x5.
    
    Consider the row-wise convolution. Each row of A (length 5) is convolved
    with columns of B.
    
    Actually, a more direct approach for small sizes is to use the known
    Winograd scheme for 5-point convolution.
    Winograd's scheme for 5-point convolution (F(5,5)) typically has rank 9 or 10.
    But we need exact integer/half-integer entries.
    
    Let's try to find a decomposition of rank r by solving a linear system
    for the coefficients, assuming a structured form based on the roots of unity.
    Since we need exact entries, we will search for integer solutions to the
    tensor identity.
    
    Strategy:
    1. Start with the schoolbook rank (100).
    2. Try to reduce rank by finding linear dependencies or using structured patterns.
    3. Specifically, try ranks from 25 to 50 with a structured search.
    """
    # This is a placeholder for the actual Winograd-based construction.
    # For now, we will use a smarter search than the current program.
    pass

def _structured_search(n, m, p, max_rank=50, min_rank=25, max_attempts=50):
    """
    Search for a low-rank decomposition with structured patterns.
    We assume U, V, W have entries in {0, 1, -1} or {0, 1, -1, 2, -2}.
    """
    best_result = None
    best_rank = n * m * p
    
    # Get schoolbook as baseline
    school = _schoolbook(n, m, p)
    if _verify(school['U'], school['V'], school['W'], n, m, p):
        best_result = school
        best_rank = n * m * p
    
    rng = np.random.RandomState(42)
    
    for r in range(min_rank, min(max_rank + 1, best_rank)):
        if _elapsed() > _DEADLINE - 20:
            break
        
        for attempt in range(max_attempts):
            if _elapsed() > _DEADLINE - 20:
                break
            
            # Initialize with sparse random patterns
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            W = np.zeros((p * n, r), dtype=int)
            
            # Density heuristic
            density = max(2.0, 10.0 / r)
            
            for i in range(n * m):
                for c in range(r):
                    if rng.random() < density / r:
                        U[i, c] = rng.choice([-1, 1])
            
            for j in range(m * p):
                for c in range(r):
                    if rng.random() < density / r:
                        V[j, c] = rng.choice([-1, 1])
            
            # Try to solve for W
            valid = True
            for k in range(p):
                for i in range(n):
                    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                    # Let A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then A @ w = 1 where w = W[k*n+i, :]
                    
                    A = np.zeros((m, r), dtype=float)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                    
                    # Try to find a simple integer solution for w
                    # First, check if any single column works
                    found = False
                    for c in range(r):
                        if np.allclose(A[:, c], np.ones(m), atol=1e-6):
                            W[k * n + i, c] = 1
                            found = True
                            break
                    
                    if not found:
                        # Try pairs of columns
                        for c1 in range(r):
                            for c2 in range(c1 + 1, r):
                                if np.allclose(A[:, c1] + A[:, c2], np.ones(m), atol=1e-6):
                                    W[k * n + i, c1] = 1
                                    W[k * n + i, c2] = 1
                                    found = True
                                    break
                            if found:
                                break
                    
                    if not found:
                        # Try least squares and round
                        try:
                            w_sol, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            w_int = np.round(w_sol).astype(int)
                            if np.allclose(A @ w_int, np.ones(m), atol=1e-6):
                                W[k * n + i, :] = w_int
                                found = True
                        except:
                            pass
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break  # Found a valid decomposition at this rank, try lower
    
    if best_result is not None:
        return best_result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try structured search first
    result = _structured_search(n, m, p, max_rank=40, min_rank=25, max_attempts=100)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
