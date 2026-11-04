"""
Structural change: search for rank-minimal exact tensor decompositions of the
matrix multiplication tensor M_{3,3,7} using an integer-constrained optimization
approach with a deadline guard.

Strategy:
  1.  The tensor T has entries T[i,j,k] = 1 if and only if (i, j, k) is a valid
      multiplication index triple. For a 3x3 * 3x7 multiplication, the valid
      triples are (i, j, k) where i in [0,2], j in [0,2], k in [0,6], and the
      contribution to output (i, k) comes from input (i, j) * (j, k).
      
      The tensor is rank-1 decomposed as T = sum_c u_c ⊗ v_c ⊗ w_c.
      We need to find minimal rank R such that there exist U (9xR), V (21xR),
      W (21xR) with entries in {0, ±1/2, ±1} (or more generally, rational/integer)
      satisfying the exact tensor equation.
      
  2.  We use a randomized local search / simulated annealing approach:
      - Initialize U, V, W with sparse random integer/half-integer entries.
      - Iteratively modify entries (set to 0, ±1/2, ±1) and accept changes that
        reduce the residual ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2.
      - Once residual is zero (or very small), verify exactness.
      
  3.  We try ranks from a lower bound upward. The trivial rank is 9*21 = 189.
      Known lower bound for M_{3,3,3} is 19, so for M_{3,3,7} a reasonable
      starting point is around 30-50.
      
  4.  Deadline guard: track elapsed time and return best result found before
      the 150-second wall-clock limit.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# Deadline guard
START_TIME = time.time()
DEADLINE_SECONDS = 145  # safety margin before 150s

def _elapsed():
    return time.time() - START_TIME

def _deadline_exceeded():
    return _elapsed() > DEADLINE_SECONDS

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n, m, p) flattened.
    T[i, j, k] = 1 if (i, j, k) is a valid index triple.
    For M_{n,m,p}, the tensor has entries T[i,j,k] = 1 for all valid (i,j,k).
    
    Actually, the tensor for matrix multiplication C = A*B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is because each triple (i,j,k) contributes to the multiplication.
    
    The factor matrices are:
    U[i*m + j, c] - input A factor
    V[j*p + k, c] - input B factor  
    W[k*n + i, c] - output C factor (trace convention: W[k*n + i, r])
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    """
    # The tensor is all ones for valid indices
    # Shape: (n, m, p) with all entries = 1
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _tensor_residual(U, V, W, n, m, p):
    """
    Compute the residual ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    # Compute the reconstructed tensor
    # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                diff = 1.0 - val
                residual += diff * diff
    return residual

def _tensor_residual_fast(U, V, W, n, m, p):
    """
    Fast residual computation using matrix operations.
    For each c, the contribution is u_c ⊗ v_c ⊗ w_c.
    We can compute this more efficiently.
    """
    R = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    # Reconstruct tensor
    # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(R):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Outer product structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.sum((T - T_recon) ** 2)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(R):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.max(np.abs(T - T_recon)) < tol

def _random_sparse_matrix(rows, rank, allowed_values, density=0.3):
    """Generate a random sparse matrix with entries from allowed_values."""
    M = np.zeros((rows, rank), dtype=np.float64)
    for r in range(rows):
        for c in range(rank):
            if random.random() < density:
                M[r, c] = random.choice(allowed_values)
    return M

def _local_search(R, n, m, p, max_iters=50000, seed=None):
    """
    Local search to find a rank-R decomposition.
    Returns (U, V, W, residual) or None if not found.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    allowed_values = [0, 0.5, 1, -0.5, -1]
    density = 0.25
    
    U = _random_sparse_matrix(n*m, R, allowed_values, density)
    V = _random_sparse_matrix(m*p, R, allowed_values, density)
    W = _random_sparse_matrix(p*n, R, allowed_values, density)
    
    current_res = _tensor_residual_fast(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_res = current_res
    
    temp = 0.1
    cooling = 0.999
    
    for it in range(max_iters):
        if _deadline_exceeded():
            break
        if current_res < 1e-12:
            # Found exact solution
            break
        
        # Random perturbation
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            r = random.randint(0, n*m - 1)
            c = random.randint(0, R - 1)
            old_val = U[r, c]
            new_val = random.choice(allowed_values)
            U[r, c] = new_val
            new_res = _tensor_residual_fast(U, V, W, n, m, p)
            if new_res < current_res or random.random() < np.exp(-(new_res - current_res) / temp):
                current_res = new_res
            else:
                U[r, c] = old_val
        elif mat == 'V':
            r = random.randint(0, m*p - 1)
            c = random.randint(0, R - 1)
            old_val = V[r, c]
            new_val = random.choice(allowed_values)
            V[r, c] = new_val
            new_res = _tensor_residual_fast(U, V, W, n, m, p)
            if new_res < current_res or random.random() < np.exp(-(new_res - current_res) / temp):
                current_res = new_res
            else:
                V[r, c] = old_val
        else:
            r = random.randint(0, p*n - 1)
            c = random.randint(0, R - 1)
            old_val = W[r, c]
            new_val = random.choice(allowed_values)
            W[r, c] = new_val
            new_res = _tensor_residual_fast(U, V, W, n, m, p)
            if new_res < current_res or random.random() < np.exp(-(new_res - current_res) / temp):
                current_res = new_res
            else:
                W[r, c] = old_val
        
        if current_res < best_res:
            best_res = current_res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        temp *= cooling
    
    if best_res < 1e-10:
        return best_U, best_V, best_W, best_res
    return None

def construct():
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
    # This is the schoolbook solution with rank n*m*p = 63
    # Try to find a better one
    
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But realistically, for matrix multiplication, the rank is at least the
    # tensor rank which is unknown but for small cases can be computed.
    
    # Try ranks from 20 upward
    best_result = None
    for R in range(20, min(64, n*m*p + 1)):
        if _deadline_exceeded():
            break
        
        # Try multiple random restarts
        for attempt in range(5):
            if _deadline_exceeded():
                break
            
            result = _local_search(R, n, m, p, max_iters=20000, seed=R*1000+attempt)
            if result is not None:
                U_arr, V_arr, W_arr, res = result
                if _verify_exact(U_arr, V_arr, W_arr, n, m, p):
                    # Convert to list of lists
                    U_list = U_arr.tolist()
                    V_list = V_arr.tolist()
                    W_list = W_arr.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
    
    # If no better solution found, return schoolbook
    return {"U": U, "V": V, "W": W}

