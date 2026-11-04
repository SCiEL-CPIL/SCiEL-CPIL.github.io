<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _construct_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_strassen_like(n, m, p):
    """
    Try to find a lower-rank decomposition by combining terms.
    For matrix multiplication, we can try to find linear dependencies.
    """
    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook()
    
    # Try various rank reductions
    # The theoretical minimum for 3x5x7 is not known, but we can try to find
    # a decomposition better than 105.
    
    # Strategy: Start with schoolbook and try to find linear combinations
    # that allow us to reduce the rank.
    
    # For now, return schoolbook as baseline
    return schoolbook

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _construct_schoolbook()
    
    # Try to improve using tensor decomposition search
    # The key is to find if there are linear dependencies in the schoolbook
    # decomposition that allow rank reduction.
    
    # For 3x5x7, the rank is at most 105 (schoolbook).
    # Known lower bounds: For m x n x p matrix multiplication,
    # the rank is at least max(m*n, n*p, m*p) = max(15, 35, 21) = 35.
    # But actual minimum is likely much higher.
    
    # Try a systematic search for lower rank decompositions
    # using the fact that we need T[i,j,k] = 1 for all valid (i,j,k)
    
    # Represent the problem as: find U, V, W with minimal rank such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is equivalent to finding a 3-tensor decomposition.
    
    # For small dimensions, we can try to find decompositions using
    # algebraic methods.
    
    # One approach: Try to find if the tensor has a low-rank structure
    # by looking at slices.
    
    # Let's try a simple heuristic: see if we can combine some of the
    # schoolbook terms.
    
    # For now, just return schoolbook as we don't have a guaranteed
    # improvement in the time budget.
    return best
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(start_rank, n, m, p, max_time=100):
    """
    Try to find a decomposition with rank < start_rank.
    Uses iterative approach: fix two matrices, solve for the third.
    """
    start_time = time.time()
    best_rank = start_rank
    best_result = _construct_schoolbook()
    
    # Try ranks from start_rank-1 down
    for target_rank in range(start_rank - 1, max(start_rank - 20, 34), -1):
        if time.time() - start_time > max_time:
            break
        
        # Try to find a decomposition of rank target_rank
        # Strategy: Initialize with random matrices and use alternating least squares
        for attempt in range(5):
            if time.time() - start_time > max_time:
                break
            
            np.random.seed(42 + attempt)
            
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() - start_time > max_time:
                    break
                
                # Solve for W: for each (k,i), we need
                # sum_c W[k*n+i, c] * (sum over j of U[i*m+j,c]*V[j*p+k,c]) = 1
                # This is: W_row * M_row = 1 where M_row[c] = sum_j U[i*m+j,c]*V[j*p+k,c]
                # But this is underdetermined. Instead, use least squares.
                
                # Build system: for each (i,j,k), 
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Fix U and V, solve for W.
                # For each column c, W[:,c] appears in terms where U and V have entries in column c.
                
                # Alternative: use the fact that this is a 3-way tensor.
                # Let's use a different approach: project and solve.
                
                # For fixed U and V, the equation is:
                # For each (i,k): sum_c W[k*n+i,c] * B_{i,k,c} = 1
                # where B_{i,k,c} = sum_j U[i*m+j,c] * V[j*p+k,c]
                # This gives n*p equations in n*p*target_rank unknowns.
                # We can set up a least squares problem.
                
                # Actually, let's use a simpler approach: 
                # The tensor T has entries T[i,j,k] = 1.
                # We want T = sum_c u_c ⊗ v_c ⊗ w_c
                # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
                # with the specific index structure.
                
                # For fixed U and V, we can solve for W using least squares.
                # The equation for entry (i,j,k) is:
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # 
                # Group by (k,i): for fixed (k,i), we have m equations (one for each j):
                # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for j=0,...,m-1
                # 
                # This is: W[k*n+i,:] * A_{k,i} = 1_vector
                # where A_{k,i}[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # 
                # So for each (k,i), we have a system: A_{k,i} * W[k*n+i,:]^T = 1
                # This is m equations in target_rank unknowns.
                
                # Build the full system for W
                num_eq = n * m * p
                num_unknowns = p * n * target_rank
                
                # This is getting complex. Let me use a different strategy.
                
                # Strategy: Use the structure to do alternating updates.
                # For each term c, we have vectors u_c, v_c, w_c.
                # The constraint is: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
                
                # Let's try a simpler heuristic: start from schoolbook and 
                # try to merge terms.
                pass
            
            # If alternating minimization doesn't work, try a different approach
            break
        
        # Try a more direct approach: use linear algebra to find dependencies
        # For rank r, we need 3*r free parameters (up to scaling) satisfying 105 equations.
        # This is a system of 105 polynomial equations in 3*r variables.
        
        # For now, let's try a greedy approach: start with schoolbook and try to 
        # find pairs of terms that can be merged.
        if time.time() - start_time > max_time:
            break
    
    return best_rank, best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _construct_schoolbook()
    best_rank = n * m * p
    
    # Try to improve using tensor decomposition search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        rank, result = _try_rank_reduction(best_rank, n, m, p, max_time=remaining_time - 5)
        if rank < best_rank and _verify(result['U'], result['V'], result['W'], n, m, p):
            best = result
            best_rank = rank
    
    return best
>>>>>>> REPLACE

Wait, I realize my approach has issues. Let me rethink this more carefully. The alternating minimization approach I sketched is not quite right. Let me implement a cleaner version that actually works.

For a given rank `r`, the problem is: find matrices U (n*m × r), V (m*p × r), W (p*n × r) such that for all (i,j,k):
`sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1`

This is a system of n*m*p = 105 equations in 3*r unknowns. For r < 105, this is underdetermined, so solutions should exist if we can find them.

A practical approach: fix U and V, then solve for W using least squares. Then fix U and W, solve for V. Then fix V and W, solve for U. Iterate.

Let me implement this properly:

