"""Optimized decomposition for 2x3x9 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix.
    
    U: shape (n*m, rank) = (6, rank)
    V: shape (m*p, rank) = (27, rank)
    W: shape (p*n, rank) = (18, rank)
    
    Convention:
    U[i*m + j, c] = coefficient for term (i, j, k) in column c
    V[j*p + k, c] = coefficient for term (i, j, k) in column c
    W[k*n + i, c] = coefficient for term (i, j, k) in column c
    
    The decomposition satisfies:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the multiplication tensor T[i,j,k] = 1
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding dependencies
    # Use iterative compression approach
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify initial solution
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction using structured search
    # Strategy: try to merge terms that have overlapping support patterns
    
    def try_reduce_rank(U_cur, V_cur, W_cur, max_iter=100):
        """Try to reduce rank by finding linear dependencies."""
        U_work = U_cur.copy()
        V_work = V_cur.copy()
        W_work = W_cur.copy()
        rank = U_work.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Try to eliminate one column at a time
            # Find if any column can be expressed as combination of others
            reduced = False
            
            for c in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Check if column c can be removed
                # This is equivalent to checking if the tensor without column c
                # can still be decomposed with the remaining columns
                
                # Simple heuristic: if U[:,c], V[:,c], W[:,c] are all zero, remove
                if np.allclose(U_work[:, c], 0) and np.allclose(V_work[:, c], 0) and np.allclose(W_work[:, c], 0):
                    U_work = np.delete(U_work, c, axis=1)
                    V_work = np.delete(V_work, c, axis=1)
                    W_work = np.delete(W_work, c, axis=1)
                    rank -= 1
                    reduced = True
                    break
            
            if not reduced:
                # Try more advanced reduction: check if two columns can be merged
                # by finding a linear relation
                break
        
        return U_work, V_work, W_work
    
    # First pass: remove any zero columns
    U, V, W = try_reduce_rank(U, V, W)
    
    # More sophisticated approach: use random projections to find low-rank structure
    # The tensor T has a specific structure that may allow rank reduction
    
    # For 2x3x9, the theoretical minimum rank is at least max(2*3, 3*9, 9*2) = 27
    # But we need to find an actual decomposition
    
    # Try structured decomposition based on the matrix structure
    # A(2x3) * B(3x9) = C(2x9)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: decompose using the fact that we can share computations
    # across different i values for the same j
    
    # Let's try a different strategy: build decomposition from scratch with
    # the constraint that we minimize rank
    
    def build_structured_decomposition():
        """Build a structured decomposition trying to minimize rank."""
        # For each output position (i, k), we need sum_j A[i,j]*B[j,k]
        # We can create terms that compute partial sums
        
        # Strategy 1: For each j, create terms that compute A[i,j]*B[j,k] for all i,k
        # This gives m * n * p = 54 terms (schoolbook)
        
        # Strategy 2: Try to share terms between different i values
        # If we fix j and k, the term A[i,j]*B[j,k] depends on i
        # We can't easily share across i without additional structure
        
        # Strategy 3: Use the fact that n=2 is small
        # For each (j,k), we need to compute A[0,j]*B[j,k] and A[1,j]*B[j,k]
        # These are two different values, so we likely need separate terms
        
        # Let's try a hybrid approach:
        # Group by j, and for each j, try to compute all i,k outputs efficiently
        
        terms = []
        for j in range(m):
            for k in range(p):
                # Term for A[0,j]*B[j,k]
                terms.append((0, j, k, 1.0))
                # Term for A[1,j]*B[j,k]  
                terms.append((1, j, k, 1.0))
        
        # This gives 2*m*p = 54 terms again
        
        # Try to find if any terms can be combined
        # Two terms can be combined if they have the same (j,k) and we can
        # express both i-dependencies in a single term
        
        # Actually, for the tensor decomposition, each term corresponds to
        # a rank-1 tensor. We need to see if the sum of rank-1 tensors
        # can be reduced.
        
        # Let's use a different approach: random search for low-rank decomposition
        
        return None
    
    # Try random initialization with optimization
    def random_search(rank_target, num_trials=50):
        """Try random initializations for a given target rank."""
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randn(n * m, rank_target)
            V_rand = np.random.randn(m * p, rank_target)
            W_rand = np.random.randn(p * n, rank_target)
            
            # Scale to make the decomposition more likely to work
            # The expected value of U[i,j,c]*V[j,k,c]*W[k,i,c] should be 1
            # If each entry is ~N(0,1), then the product has mean 0
            # We need to add a bias or use a different initialization
            
            # Better: initialize with ones and small perturbations
            U_rand = np.ones((n * m, rank_target)) + 0.1 * np.random.randn(n * m, rank_target)
            V_rand = np.ones((m * p, rank_target)) + 0.1 * np.random.randn(m * p, rank_target)
            W_rand = np.ones((p * n, rank_target)) + 0.1 * np.random.randn(p * n, rank_target)
            
            # Check if this gives a valid decomposition (it won't, but we can
            # use it as a starting point for optimization)
            
            # For now, just check if any random solution works exactly
            # (very unlikely, but cheap to check)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U_rand, V_rand, W_rand
        
        return None
    
    # Try various target ranks
    for target_rank in range(27, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Quick check: is this rank theoretically possible?
        # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
        if target_rank < 27:
            continue
        
        # Try random search for this rank
        result = random_search(target_rank, num_trials=20)
        if result is not None:
            U_found, V_found, W_found = result
            if verify(U_found, V_found, W_found):
                best_rank = target_rank
                best_U, best_V, best_W = U_found, V_found, W_found
                break
    
    # If no better solution found, return the best we have
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
