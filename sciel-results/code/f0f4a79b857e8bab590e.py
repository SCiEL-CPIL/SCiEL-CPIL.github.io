import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = len(best['U'][0])
    
    # Try to find a lower-rank decomposition
    # Strategy: search for sparse support patterns over GF(2) or GF(3),
    # then lift to rationals
    for trial_rank in range(80, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        result = _try_rank_decomposition(n, m, p, trial_rank)
        if result is not None:
            best = result
            best_rank = trial_rank
            break
    
    return best

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _try_rank_decomposition(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition using:
    1. Random sparse support patterns
    2. Solve linear system for coefficients over rationals
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return None
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Number of constraints: for each (i,j,k,r) where i!=r or j!=k, the sum must be 0
    # And for each (i,j,k) with r=i, the sum must be 1
    # Total constraints: n*m*p*n - n*m*p (zero constraints) + n*m*p (one constraints)
    # = n*m*p*(n-1) + n*m*p = n*m*p*n
    
    # We have 3 * target_rank * (U_rows + V_rows + W_rows) / 3 variables... no
    # Variables: U has U_rows * target_rank, V has V_rows * target_rank, W has W_rows * target_rank
    # Total vars = target_rank * (U_rows + V_rows + W_rows)
    
    # For the bilinear constraints, we need to be more careful
    # The tensor is T[i,j,k,r] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+r,c]
    # This is trilinear, not linear in the variables.
    # So we can't directly solve a linear system.
    
    # Instead, use a different approach: fix two matrices and solve for the third
    # Or use alternating minimization
    
    # Approach: Use random initialization with sparsity, then alternating least squares
    # with exact rational arithmetic where possible
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
        
        # Generate sparse random factors
        density = 0.3
        U = _random_sparse_matrix(U_rows, target_rank, density)
        V = _random_sparse_matrix(V_rows, target_rank, density)
        W = _random_sparse_matrix(W_rows, target_rank, density)
        
        # Try to refine using alternating minimization
        result = _refine_decomposition(U, V, W, n, m, p, target_rank)
        if result is not None:
            if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                return result
    
    return None

def _random_sparse_matrix(rows, cols, density):
    """Generate a sparse random matrix with small integer entries."""
    M = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if random.random() < density:
                M[i][j] = random.choice([-2, -1, 1, 2])
    return M

def _refine_decomposition(U, V, W, n, m, p, rank):
    """
    Alternating minimization: fix two factors, solve for the third.
    Since the problem is trilinear, we use a nonlinear approach.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We'll use a gradient-free approach: coordinate descent on sparsity patterns
    # and local search on values
    
    best_error = _compute_error(U, V, W, n, m, p)
    
    for iteration in range(20):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        
        # Try perturbing U
        for _ in range(10):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            U_pert = [row[:] for row in U]
            i = random.randint(0, U_rows - 1)
            j = random.randint(0, rank - 1)
            U_pert[i][j] += random.choice([-2, -1, 1, 2])
            err = _compute_error(U_pert, V, W, n, m, p)
            if err < best_error:
                U = U_pert
                best_error = err
                improved = True
        
        # Try perturbing V
        for _ in range(10):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            V_pert = [row[:] for row in V]
            i = random.randint(0, V_rows - 1)
            j = random.randint(0, rank - 1)
            V_pert[i][j] += random.choice([-2, -1, 1, 2])
            err = _compute_error(U, V_pert, W, n, m, p)
            if err < best_error:
                V = V_pert
                best_error = err
                improved = True
        
        # Try perturbing W
        for _ in range(10):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            W_pert = [row[:] for row in W]
            i = random.randint(0, W_rows - 1)
            j = random.randint(0, rank - 1)
            W_pert[i][j] += random.choice([-2, -1, 1, 2])
            err = _compute_error(U, V, W_pert, n, m, p)
            if err < best_error:
                W = W_pert
                best_error = err
                improved = True
        
        if not improved:
            break
        
        if best_error == 0:
            break
    
    if best_error == 0:
        return {"U": U, "V": V, "W": W}
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    rank = len(U[0])
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1 if (i == r and j == k) else 0
                    error += (s - expected) ** 2
    return error

# EVOLVE-BLOCK-END
