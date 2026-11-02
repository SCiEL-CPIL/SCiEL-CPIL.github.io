"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # We decompose T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] ? 
    # No, indices are:
    # U has shape (n, R) ? No, standard is U[i,j,r], V[j,k,r], W[k,i,r] or similar.
    # Task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # My insight:
    # For fixed i, k:
    # sum_r (U[i, r] * V[j, r]) * W[k, i, r] = 1 for all j?
    # Wait, U index is i*m+j. So U depends on j.
    # So my previous simplification was slightly off.
    # U[i*m+j, r] depends on j.
    # So for fixed i, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This is a system for W[k*n+i, r] where the coefficients depend on j.
    # Let w_{i,k} be the vector of W entries for row k*n+i.
    # Equation for j: sum_r (U[i*m+j, r] V[j*p+k, r]) w_{i,k}[r] = 1.
    # This is a linear system for w_{i,k} for each (i,k) pair.
    # Matrix M_{i,k} of size m x R. M_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r].
    # We need M_{i,k} w_{i,k} = 1.
    
    def solve_for_W(U, V, R):
        """Given U, V, solve for W. Returns None if no integer/half-integer solution."""
        W = [[0] * R for _ in range(p * n)]
        for i in range(n):
            for k in range(p):
                # Construct M_{i,k}
                M = np.zeros((m, R), dtype=float)
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i * m + j][r] * V[j * p + k][r]
                
                # Solve M w = 1
                # Check if 1 is in column space
                # Use least squares
                try:
                    w, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check if solution is close to integer/half-integer
                # And if M w is exactly 1
                # We require exact integer/half-integer entries.
                valid = True
                for r in range(R):
                    val = w[r]
                    # Check if val is close to integer or half-integer
                    # Multiply by 2 to check integer
                    val2 = val * 2
                    nearest = round(val2)
                    if abs(val2 - nearest) > 1e-9:
                        valid = False
                        break
                if not valid:
                    return None
                
                # Verify exactness
                for j in range(m):
                    total = 0
                    for r in range(R):
                        total += M[j, r] * (nearest / 2) # Wait, nearest is val2
                        # w[r] = nearest / 2
                    if abs(total - 1.0) > 1e-9:
                        return None
                
                # Store W
                for r in range(R):
                    W[k * n + i][r] = w[r]
        
        return W

    # Search for U, V
    import random
    random.seed(42)
    
    best_rank = 24
    best_solution = None
    
    # Try ranks from 22 to 23
    for R in [22, 23]:
        attempts = 0
        max_attempts = 50000
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            # Generate sparse U, V
            # U has n*m = 6 rows, R cols
            # V has m*p = 12 rows, R cols
            # Use entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
            
            W = solve_for_W(U, V, R)
            if W is not None:
                best_rank = R
                best_solution = (U, V, W)
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
