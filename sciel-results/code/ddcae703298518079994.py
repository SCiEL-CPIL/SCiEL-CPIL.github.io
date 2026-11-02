"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
We attempt to find a lower-rank decomposition than the schoolbook rank of 100.
Strategy:
1. Start with the schoolbook decomposition (rank 100).
2. Use a structured search for lower-rank decompositions by leveraging
   the bilinear nature of matrix multiplication.
3. Specifically, we model the problem as finding a set of rank-1 tensors
   (triads) that sum to the multiplication tensor.
4. We use a combination of:
   - Known structured decompositions (e.g., based on smaller matrix multiplication)
   - Random search with verification
   - ALS (Alternating Least Squares) optimization for continuous relaxation

The key insight is that the rank of M_{4,5,5} is bounded below by max(20, 25, 20) = 25
and above by 100 (schoolbook). We search for decompositions with rank < 100.

For 4x5 * 5x5, we can potentially exploit structure from smaller matrix multiplications.
For example, if we can find a rank-r decomposition for smaller cases, we might extend it.

However, the most reliable approach within the time budget is:
1. Use the schoolbook decomposition as a fallback.
2. Try random sparse decompositions with verification.
3. If found, return the best (lowest rank) valid decomposition.
"""
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form a valid tensor decomposition for matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want T[i,j,k] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_random_sparse(n, m, p, rank, seed=0, sparsity=0.2):
    """Try a random sparse decomposition of given rank."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n * m):
        for c in range(rank):
            if rng.random() < sparsity:
                U[i, c] = rng.choice([-1, 1])
    
    for j in range(m * p):
        for c in range(rank):
            if rng.random() < sparsity:
                V[j, c] = rng.choice([-1, 1])
    
    for k in range(p * n):
        for c in range(rank):
            if rng.random() < sparsity:
                W[k, c] = rng.choice([-1, 1])
    
    return U.tolist(), V.tolist(), W.tolist()

def _als_optimize(n, m, p, rank, iters=20, seed=0, tol=1e-4):
    """
    Attempt to find a decomposition of given rank using ALS on the continuous relaxation.
    We minimize ||T - sum_c U_c V_c W_c||^2 where T is the all-ones tensor.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(iters):
        # Update U: For each (i,j), solve least squares
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Let M[i,j] be a (p x rank) matrix where M[k, c] = V[j,k,c] * W[k,i,c]
        # Then T[i,j,:] = U[i,j,:] @ M.T ? No.
        # T[i,j,k] = sum_c U[i,j,c] * M[c, k] where M[c,k] = V[j,k,c] * W[k,i,c]
        # So T[i,j,:] = U[i,j,:] @ M where M is (rank x p)
        
        for i in range(n):
            for j in range(m):
                # Construct M of shape (rank, p)
                # M[c, k] = V[j, k, c] * W[k, i, c]
                M = np.zeros((rank, p))
                for c in range(rank):
                    for k in range(p):
                        M[c, k] = V[j * p + k][c] * W[k * n + i][c]
                
                # Solve U[i,j,:] @ M = ones(p)
                # This is a least squares problem: U[i,j,:] @ M = 1
                # Equivalent to M.T @ U[i,j,:].T = 1
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: For each (j,k), solve least squares
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed (j,k), T[:,j,k] = sum_c (U[:,j,c] * W[k,:,c]) * V[j,k,c]
        # Let N be (n x rank) where N[i, c] = U[i*m+j, c] * W[k*n+i, c]
        # Then T[:,j,k] = N @ V[j,k,:].T
        # So N @ V[j,k,:].T = ones(n)
        
        for j in range(m):
            for k in range(p):
                N = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        N[i, c] = U[i * m + j][c] * W[k * n + i][c]
                
                try:
                    sol, residuals, rank_n, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: For each (k,i), solve least squares
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed (k,i), T[i,:,k] = sum_c (U[i,:,c] * V[:,k,c]) * W[k,i,c]
        # Let P be (m x rank) where P[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # Then T[i,:,k] = P @ W[k,i,:].T
        # So P @ W[k,i,:].T = ones(m)
        
        for k in range(p):
            for i in range(n):
                P = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        P[j, c] = U[i * m + j][c] * V[j * p + k][c]
                
                try:
                    sol, residuals, rank_p, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol
                except np.linalg.LinAlgError:
                    pass
    
    # Round to nearest integer or half-integer if close
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Check if the rounded version is valid
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-3):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Check if the original is close enough
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 4x5 * 5x5 matrix multiplication.
    
    U has shape (n*m, rank) = (20, rank)
    V has shape (m*p, rank) = (25, rank)
    W has shape (p*n, rank) = (20, rank)
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0, n), j in [0, m), k in [0, p).
    """
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 100
    
    # Try to find lower-rank decompositions
    # Known lower bound: max(n*m, m*p, n*p) = 25
    # We'll try ranks from 25 to 99
    
    random.seed(42)
    np.random.seed(42)
    
    # First, try some structured approaches
    # For 4x5 * 5x5, we can think of it as blocks
    
    # Try random sparse search for various ranks
    for target_rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random seeds for this rank
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U, V, W = _try_random_sparse(n, m, p, target_rank, seed=seed, sparsity=0.3)
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                if best_rank <= 25:
                    return best
        
        # Try ALS for this rank
        if time.time() < deadline - 5:
            result = _als_optimize(n, m, p, target_rank, iters=10, seed=seed)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                if best_rank <= 25:
                    return best
    
    return best
