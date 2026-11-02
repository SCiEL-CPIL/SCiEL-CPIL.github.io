"""Program for matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    We use an alternating least squares approach with NNLS, trying ranks from low to high.
    The key insight is that for fixed V and W, each row of U is determined by a linear system.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 130  # Leave 20s safety margin
    
    def validate(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, max_iters=500, seed=0):
        """
        Try to find a valid decomposition of given rank using alternating NNLS.
        """
        rng = random.Random(seed)
        
        # Initialize with random values
        U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve A * U_row = ones(p)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        sol, _ = nnls(A, b)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass
            
            # Update V: for each (j,k), solve A * V_row = ones(n)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, _ = nnls(A, b)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass
            
            # Update W: for each (k,i), solve A * W_row = ones(m)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, _ = nnls(A, b)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if err < 1e-10:
                # Found a solution! Try rounding to half-integers
                for rounding_factor in [2, 1]:
                    U_r = [[round(x * rounding_factor) / rounding_factor for x in row] for row in U]
                    V_r = [[round(x * rounding_factor) / rounding_factor for x in row] for row in V]
                    W_r = [[round(x * rounding_factor) / rounding_factor for x in row] for row in W]
                    
                    if validate(U_r, V_r, W_r, tol=1e-9):
                        return U_r, V_r, W_r
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound considerations suggest starting from around 10-12
    for rank in range(10, 31):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, max_iters=300, seed=rank * 1000 + attempt)
            if result is not None:
                U, V, W = result
                if validate(U, V, W, tol=1e-9):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook decomposition (rank 30)
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
# EVOLVE-BLOCK-END
