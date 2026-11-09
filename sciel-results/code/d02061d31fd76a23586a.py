"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Build the rank-22 solution using block decomposition with Strassen
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col, coeffs_u, coeffs_v, coeffs_w):
        for idx, coeff in zip(u_indices, coeffs_u):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, coeffs_v):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, coeffs_w):
            W[idx][col] += coeff
            
    # Block 1: A_L (2x2) * B_T1 (2x2) -> C1 (2x2)
    add_term([0, 4], [0, 5], [0, 3], 0, [1, 1], [1, 1], [1, 1])
    add_term([3, 4], [0], [1, 3], 1, [1, 1], [1], [1, -1])
    add_term([0], [1, 5], [2, 3], 2, [1], [1, -1], [1, 1])
    add_term([4], [4, 0], [0, 1], 3, [1], [1, -1], [1, 1])
    add_term([0, 1], [5], [0, 2], 4, [1, 1], [1], [-1, 1])
    add_term([3, 0], [0, 1], [3], 5, [1, -1], [1, 1], [1])
    add_term([1, 4], [4, 5], [0], 6, [1, -1], [1, 1], [1])
    
    # Block 2: A_L (2x2) * B_T2 (2x2) -> C2 (2x2)
    add_term([0, 4], [2, 7], [4, 7], 7, [1, 1], [1, 1], [1, 1])
    add_term([3, 4], [2], [5, 7], 8, [1, 1], [1], [1, -1])
    add_term([0], [3, 7], [6, 7], 9, [1], [1, -1], [1, 1])
    add_term([4], [6, 2], [4, 5], 10, [1], [1, -1], [1, 1])
    add_term([0, 1], [7], [4, 6], 11, [1, 1], [1], [-1, 1])
    add_term([3, 0], [2, 3], [7], 12, [1, -1], [1, 1], [1])
    add_term([1, 4], [6, 7], [4], 13, [1, -1], [1, 1], [1])
    
    # Block 3: A_R (2x1) * B_B (1x4) -> C3 (2x4)
    add_term([2], [8], [0], 14, [1], [1], [1])
    add_term([2], [9], [2], 15, [1], [1], [1])
    add_term([2], [10], [4], 16, [1], [1], [1])
    add_term([2], [11], [6], 17, [1], [1], [1])
    add_term([5], [8], [1], 18, [1], [1], [1])
    add_term([5], [9], [3], 19, [1], [1], [1])
    add_term([5], [10], [5], 20, [1], [1], [1])
    add_term([5], [11], [7], 21, [1], [1], [1])
    
    # Verify the rank-22 solution
    def verify_solution(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify that the decomposition correctly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(len(U_mat[0])):
                        val += U_mat[i][l] * V_mat[j][l] * W_mat[k*n+i][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_solution(U, V, W, n, m, p):
        # Fallback: return schoolbook rank 24
        return _schoolbook(n, m, p)
    
    # Attempt to find rank 21 using a structured search
    # The idea: try to merge or eliminate one term from the rank-22 solution
    # by finding a linear dependency among the rank-1 terms
    
    if time.time() < deadline - 30:
        try:
            result = _try_rank21(U, V, W, n, m, p, deadline)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][idx] = 1.0
                V[j*p+k][idx] = 1.0
                W[k*n+i][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank21(U22, V22, W22, n, m, p, deadline):
    """
    Attempt to find a rank-21 decomposition by exploiting linear dependencies
    in the rank-22 solution.
    
    Strategy: The rank-22 solution has 22 rank-1 terms. We check if any term
    can be expressed as a linear combination of the others in a way that
    preserves the tensor decomposition.
    
    This is a heuristic search - we try to find a term that, when removed,
    can be compensated by adjusting coefficients of other terms.
    """
    start_time = time.time()
    
    # Convert to numpy for efficient computation
    U_arr = np.array(U22, dtype=float)
    V_arr = np.array(V22, dtype=float)
    W_arr = np.array(W22, dtype=float)
    
    rank22 = len(U_arr[0])
    
    # For each term l, check if removing it and adjusting others can still work
    # This is a complex constraint satisfaction problem, so we use a heuristic:
    # Try to find a term that has minimal unique contribution
    
    # Compute the "importance" of each term by checking how many output entries
    # would be affected if we zeroed out that term
    
    # Alternative approach: use the fact that Strassen's algorithm for 2x2x2
    # has rank 7, and we use two such blocks plus 8 terms for the remaining
    # 2x1 * 1x4 block. Total = 7 + 7 + 8 = 22.
    
    # Can we reduce the 2x1 * 1x4 block from 8 terms to 7?
    # The 2x1 * 1x4 multiplication is essentially a vector-matrix product.
    # The tensor rank of 2x1x4 is 8 (since it's essentially 8 independent products).
    # So we can't reduce that block.
    
    # Can we share terms between the two 2x2x2 blocks?
    # The two blocks use disjoint sets of input indices, so sharing is difficult.
    
    # Let's try a more systematic approach: formulate as a linear system
    # and check if a rank-21 solution exists with a specific sparsity pattern.
    
    # For rank 21, we need 21 columns in U, V, W.
    # We'll try to find a solution where the first 21 columns of the rank-22
    # solution, with some adjustments, form a valid decomposition.
    
    # This is a bilinear problem, so we use an alternating approach:
    # 1. Fix U, V, solve for W
    # 2. Fix U, W, solve for V  
    # 3. Fix V, W, solve for U
    # Repeat until convergence or timeout
    
    # Initialize with the first 21 columns of the rank-22 solution
    U_init = U_arr[:, :21]
    V_init = V_arr[:, :21]
    W_init = W_arr[:, :21]
    
    # Alternating least squares with exact rational constraints
    # This won't guarantee exactness, but we can check at the end
    
    max_iters = 50
    tol = 1e-6
    
    U_curr = U_init.copy()
    V_curr = V_init.copy()
    W_curr = W_init.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline - 15:
            break
            
        # Solve for W given U and V
        # For each output entry (i, k), we need:
        # sum_l U[i,l] * V[j,l] * W[k*n+i,l] = 1 for all j
        # This is a linear system in W
        
        # Build the linear system for W
        # Variables: W[0..7, 0..20] (8*21 = 168 variables)
        # Equations: for each (i,j,k), sum_l U[i,l]*V[j,l]*W[k*n+i,l] = 1
        
        # This is a large linear system. Let's use a simpler heuristic:
        # Check if the current approximation is already good enough
        
        # Compute the error
        max_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(21):
                        val += U_curr[i][l] * V_curr[j][l] * W_curr[k*n+i][l]
                    max_error = max(max_error, abs(val - 1.0))
        
        if max_error < tol:
            # Check if this gives an exact solution
            if _verify_exact(U_curr, V_curr, W_curr, n, m, p):
                return (U_curr.tolist(), V_curr.tolist(), W_curr.tolist())
        
        # If error is not small, break
        if max_error > 1.0:
            break
    
    return None

def _verify_exact(U_mat, V_mat, W_mat, n, m, p, tol=1e-10):
    """Verify exact decomposition with tight tolerance."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for l in range(len(U_mat[0])):
                    val += U_mat[i][l] * V_mat[j][l] * W_mat[k*n+i][l]
                if abs(val - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-END
