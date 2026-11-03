"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _validate_np(U, V, W, n, m, p):
    """Vectorized validation using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)
    T = np.einsum('ijkc,kic->ijk', UV, W3)
    
    expected = np.zeros((n, m, p), dtype=np.float64)
    for j in range(min(m, p)):
        expected[:, j, j] = 1.0
    
    return np.allclose(T, expected, atol=1e-9)


def _try_coordinate_descent(n, m, p, target_rank, max_iters=200, seed=42):
    """
    Try to find a rank-target_rank decomposition using coordinate descent.
    
    Strategy: 
    - Initialize U, V, W with structured random values
    - Iteratively fix two matrices and solve for the third
    - Check for exact satisfaction
    """
    np_rng = np.random.RandomState(seed)
    
    # Initialize
    U = np.zeros((n * m, target_rank), dtype=np.float64)
    V = np.zeros((m * p, target_rank), dtype=np.float64)
    W = np.zeros((p * n, target_rank), dtype=np.float64)
    
    # Structured initialization: start close to schoolbook pattern
    for c in range(target_rank):
        # Random but with some structure
        U[:, c] = np_rng.choice([-1, 0, 1], size=n * m)
        V[:, c] = np_rng.choice([-1, 0, 1], size=m * p)
        W[:, c] = np_rng.choice([-1, 0, 1], size=p * n)
    
    # Coordinate descent
    for it in range(max_iters):
        # Fix V and W, solve for U
        # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j,k)
        # This is linear in U for fixed V, W
        # U[i*m+j, c] = x_c, we need: for all k, sum_c x_c * V[j*p+k, c] * W[k*n+i, c] = delta(j,k)
        # This is a linear system: A_{k,c} * x_c = b_k where A_{k,c} = V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                # Build linear system for U[i*m+j, :]
                A = np.zeros((p, target_rank), dtype=np.float64)
                b = np.zeros(p, dtype=np.float64)
                for k in range(p):
                    b[k] = 1.0 if j == k else 0.0
                    for c in range(target_rank):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                
                # Solve A @ x = b using least squares
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Fix U and W, solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank), dtype=np.float64)
                b = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    b[i] = 1.0 if j == k else 0.0
                    for c in range(target_rank):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Fix U and V, solve for W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank), dtype=np.float64)
                b = np.zeros(m, dtype=np.float64)
                for j in range(m):
                    b[j] = 1.0 if j == k else 0.0
                    for c in range(target_rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check if we've converged
        if _validate_np(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Start with schoolbook (rank 45)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = 15
    # But for matmul, better bounds exist
    
    # Try various ranks from lower to higher
    # For (3,3,5), try ranks from 20 to 44
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for seed in range(50):
            if time.time() > deadline:
                break
            
            result = _try_coordinate_descent(n, m, p, target_rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best
# EVOLVE-BLOCK-END

