"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Return the 2x2x2 Strassen decomposition.
    U, V, W have shape (4, 7) each.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 0: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 1: (A[1,0] + A[1,1]) * B[0,0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Product 2: A[0,0] * (B[0,1] - B[1,1])
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 3: A[1,1] * (B[1,0] - B[0,0])
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 4: (A[0,0] + A[0,1]) * B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 5: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Product 6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _try_numerical_search(r, n, m, p, max_iters=100, seed=42, max_attempts=5):
    """
    Try to find a rank-r decomposition via numerical optimization.
    Returns (U, V, W) if successful, else None.
    """
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(max_attempts):
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Initialize with random values
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        def objective(params):
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r+m*p*r:].reshape(p*n, r)
            
            # Compute error: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                        err += (val - 1.0)**2
            return err
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < best_err:
                best_err = result.fun
                best_solution = result.x
        except Exception:
            pass
    
    if best_solution is None or best_err > 1e-8:
        return None
    
    U = best_solution[:n*m*r].reshape(n*m, r)
    V = best_solution[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = best_solution[n*m*r+m*p*r:].reshape(p*n, r)
    
    # Try to round to half-integers
    for scale in [2, 4, 8]:  # Try different granularities
        U_round = np.round(U * scale) / scale
        V_round = np.round(V * scale) / scale
        W_round = np.round(W * scale) / scale
        
        T = _make_tensor(n, m, p)
        if _verify_decomposition(U_round, V_round, W_round, T, tol=1e-9):
            return (U_round, V_round, W_round)
    
    return None

def _construct_block_decomposition():
    """
    Construct a block decomposition for 4x5x5 using Strassen where applicable.
    
    Strategy: Split the 4x5 matrix A into two 2x5 blocks (top and bottom).
    Split the 5x5 matrix B into blocks.
    
    For the 2x2x2 sub-problems, use Strassen (rank 7 instead of 8).
    For other sub-problems, use schoolbook or numerical search.
    
    The key idea: if we can decompose the problem into independent sub-problems,
    the total rank is the sum of the ranks of the sub-problems.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Split A (4x5) into two 2x5 blocks: A_top (rows 0-1), A_bottom (rows 2-3)
    # Split B (5x5) - we need to think about how to split this
    # Split C (4x5) into two 2x5 blocks: C_top (rows 0-1), C_bottom (rows 2-3)
    
    # C_top = A_top * B
    # C_bottom = A_bottom * B
    
    # For A_top * B (2x5 * 5x5 = 2x5):
    # We can split B into 2x2, 2x2, 1x2 blocks? No, that's not clean.
    # Better: split B into 5 columns, each 5x1
    # Or: think of it as 2x5 * 5x5
    
    # Actually, let's use a different approach.
    # The standard schoolbook rank is 4*5*5 = 100.
    # We want to find a decomposition with rank < 100.
    
    # One approach: use the fact that for 2x2x2, Strassen gives rank 7 < 8.
    # If we can partition the problem into 2x2x2 sub-problems, we save rank.
    
    # For 4x5x5:
    # Split rows of A into 2+2
    # Split columns of A (and rows of B) into 2+2+1
    # Split columns of B (and rows of C) into 2+2+1
    
    # This gives us blocks:
    # A: 2x2, 2x2, 2x1 (top), 2x2, 2x2, 2x1 (bottom)
    # B: 2x2, 2x2, 2x1, 2x2, 2x2, 2x1, 1x2, 1x2, 1x1
    # C: 2x2, 2x2, 2x1, 2x2, 2x2, 2x1 (top), 2x2, 2x2, 2x1, 2x2, 2x2, 2x1 (bottom)
    
    # This is getting complex. Let's try a simpler approach:
    # Use numerical search to find a low-rank decomposition.
    
    # First, try to find a decomposition with rank slightly less than 100
    # using numerical optimization.
    
    # Try ranks from 95 down to 80
    for r in range(95, 79, -1):
        if _elapsed() > _DEADLINE - 30:
            break
        
        result = _try_numerical_search(r, n, m, p, max_iters=80, seed=r, max_attempts=3)
        if result is not None:
            U, V, W = result
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def _schoolbook_decomposition():
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {'U': U, 'V': V, 'W': W}

def construct():
    deadline = time.time() + 120 - 10  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook decomposition as fallback
    best_result = _schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank decomposition using numerical search
    result = _construct_block_decomposition()
    if result is not None:
        U_try = np.array(result['U'])
        rank_try = U_try.shape[1]
        if rank_try < best_rank:
            # Verify
            T = _make_tensor(n, m, p)
            if _verify_decomposition(U_try, np.array(result['V']), np.array(result['W']), T):
                best_result = result
                best_rank = rank_try
    
    return best_result
# EVOLVE-BLOCK-END
