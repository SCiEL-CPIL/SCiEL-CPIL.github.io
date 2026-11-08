"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3)
    - j in [0, m) = [0, 5)  
    - k in [0, p) = [0, 7)
    
    T[i, j, k] = 1 if we multiply A[i,j] * B[j,k] to contribute to C[i,k]
    
    We seek rank r such that:
    T[i, j, k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Using the official trace convention: W[k*n+i, r]
    
    Strategy: Start with schoolbook (rank 105) and try to find lower rank
    decompositions using integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    deadline = time.time() + 130  # Leave 20s safety margin for the 150s limit
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p  # 105
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the tensor T from U, V, W
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k (since each product contributes exactly once)
        
        # More precisely, the matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1
        # And we need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Key insight: For matrix multiplication, the tensor rank is known to be 
    # between O(n^omega) and n*m*p. For small dimensions, we can search.
    
    # Approach: Use the fact that matrix multiplication tensor has structure.
    # Try various ranks from low to high, attempting to find integer/half-integer solutions.
    
    # For 3x5x7, let's try a systematic approach:
    # The tensor rank for M_{n,m,p} is at most n*m*p = 105
    # Lower bounds exist, but for small cases we can try to find good decompositions.
    
    # Strategy: Try to use the structure of matrix multiplication.
    # One approach: Use the "Strassen-like" recursive structure.
    
    # Let's try ranks from a reasonable lower bound up to 105
    # Lower bound for matrix multiplication tensor rank is at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Actually, the tensor rank lower bound is more nuanced, but let's try from 35 upward.
    
    # For efficiency, we'll use a randomized approach with local optimization.
    
    def try_rank(r, max_iter=5000, seed=None):
        """Try to find a rank-r decomposition."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Initialize with random values
        # Use integer/half-integer values: multiply by 2 to work with integers
        U = rng.randint(-3, 4, size=(nm, r)) * 0.5
        V = rng.randint(-3, 4, size=(mp, r)) * 0.5
        W = rng.randint(-3, 4, size=(pn, r)) * 0.5
        
        # Gradient-free optimization using coordinate descent
        # Objective: minimize sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        def objective(U, V, W):
            total = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        total += (val - 1.0) ** 2
            return total
        
        best_obj = objective(U, V, W)
        
        # Coordinate descent: update one variable at a time
        # For each column c, and each matrix, try to improve
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try updating U
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        old_val = U[idx_u, c]
                        # Compute gradient-like update
                        # d_obj/dU[idx_u,c] = 2 * sum_{k} (val_ijk - 1) * V[j*p+k,c] * W[k*n+i,c]
                        grad = 0.0
                        for k in range(p):
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            grad += (val - 1.0) * V[idx_v, c] * W[idx_w, c]
                        # Try small perturbations
                        for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                            new_val = old_val + delta
                            U[idx_u, c] = new_val
                            new_obj = objective(U, V, W)
                            if new_obj < best_obj - 1e-12:
                                best_obj = new_obj
                                improved = True
                            else:
                                U[idx_u, c] = old_val
            
            # Try updating V
            for c in range(r):
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        old_val = V[idx_v, c]
                        grad = 0.0
                        for i in range(n):
                            idx_u = i * m + j
                            idx_w = k * n + i
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            grad += (val - 1.0) * U[idx_u, c] * W[idx_w, c]
                        for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                            new_val = old_val + delta
                            V[idx_v, c] = new_val
                            new_obj = objective(U, V, W)
                            if new_obj < best_obj - 1e-12:
                                best_obj = new_obj
                                improved = True
                            else:
                                V[idx_v, c] = old_val
            
            # Try updating W
            for c in range(r):
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        old_val = W[idx_w, c]
                        grad = 0.0
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            grad += (val - 1.0) * U[idx_u, c] * V[idx_v, c]
                        for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                            new_val = old_val + delta
                            W[idx_w, c] = new_val
                            new_obj = objective(U, V, W)
                            if new_obj < best_obj - 1e-12:
                                best_obj = new_obj
                                improved = True
                            else:
                                W[idx_w, c] = old_val
            
            if not improved and iteration > 100:
                break
        
        # Check if we found an exact solution
        if best_obj < 1e-10:
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            if verify(U, V, W):
                return U, V, W, best_obj
            else:
                return None
        
        return None
    
    # Try different ranks from low to high
    # Start from a reasonable lower bound
    # For M_{3,5,7}, known bounds suggest rank could be significantly less than 105
    # Let's try from 50 upward, but be careful about time
    
    # Actually, let's be more strategic. The schoolbook is 105.
    # Let's try a few promising ranks.
    
    ranks_to_try = []
    # Try some specific ranks that might work
    for r in range(60, 105):
        ranks_to_try.append(r)
    
    # Shuffle to try different ones
    random.seed(42)
    random.shuffle(ranks_to_try)
    
    for r in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            
            result = try_rank(r, max_iter=2000, seed=seed)
            if result is not None:
                U, V, W, obj = result
                if r < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, r
                    break
        
        # If we found a solution for this rank, try lower ranks
        if best_rank <= r:
            # Update ranks_to_try to only include lower ranks
            ranks_to_try = [x for x in ranks_to_try if x < best_rank]
            random.shuffle(ranks_to_try)
    
    # Convert to list format as required
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
