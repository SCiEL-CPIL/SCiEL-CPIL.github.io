import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 145  # seconds, safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
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

def _try_rank_random(r, n, m, p, max_attempts=1000, seed=42):
    """Try to find rank-r decomposition via random integer search."""
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        if _elapsed() > _DEADLINE - 10:
            break
            
        # Generate random sparse U, V with entries -1, 0, 1
        # Density 0.5
        U = rng.choice([-1, 0, 1], size=(n*m, r))
        V = rng.choice([-1, 0, 1], size=(m*p, r))
        
        # Check if we can solve for W
        W = np.zeros((p*n, r), dtype=int)
        valid = True
        
        for k in range(p):
            for i in range(n):
                # System: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # Let A be 5 x r matrix
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # We need to find integer w such that A w = 1
                # We can try to find a solution by setting most w_c to 0
                # and solving for a few.
                # Try all combinations of 5 columns? Too many.
                # Instead, try to find a column c where A[:, c] is close to 1?
                # Or just use a simple heuristic:
                # If there is a column c such that A[:, c] == 1, set w_c = 1.
                found = False
                for c in range(r):
                    if np.all(A[:, c] == 1):
                        W[k*n+i, c] = 1
                        found = True
                        break
                
                if not found:
                    # Try to find a linear combination of a few columns
                    # Pick 5 random columns
                    cols = rng.choice(r, size=min(5, r), replace=False)
                    A_sub = A[:, cols]
                    try:
                        # Solve A_sub w_sub = 1
                        # Use least squares or direct solve if square
                        if A_sub.shape[1] >= m:
                            # Use a simple solver
                            w_sub, residuals, rank, s = np.linalg.lstsq(A_sub, np.ones(m), rcond=None)
                            # Check if solution is integer (or close)
                            w_int = np.round(w_sub)
                            if np.allclose(A_sub @ w_int, np.ones(m), atol=1e-6):
                                for idx, c in enumerate(cols):
                                    W[k*n+i, c] = w_int[idx]
                                found = True
                    except Exception:
                        pass
                
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify the solution
            if _verify(U, V, W, n, m, p, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank solution
    # Start from rank 99 and go down
    for r in range(99, 90, -1):
        if _elapsed() > _DEADLINE - 10:
            break
        
        result = _try_rank_random(r, n, m, p, max_attempts=200, seed=r)
        
        if result is not None:
            return result
    
    # Fallback: schoolbook
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
