"""
Tensor rank minimization for 3x3 * 3x6 matrix multiplication.

Uses randomized numerical rank estimation to find a low-rank exact decomposition.
The tensor T_{i,j,k} = 1 for all i in [0,3), j in [0,3), k in [0,6) is the
matmul tensor. We look for rank r such that T can be written as sum_{c=0}^{r-1}
U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].

Strategy: Use randomized SVD / iterative refinement to find a decomposition.
We try ranks starting from a lower bound and increase until we find a valid one.
For 3x3x6, the known lower bound is around 32-46. We try to find the minimum.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # The tensor T of shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We want to decompose this as sum_c U[:,c] * V[:,c] * W[:,c] (outer product sum)
    # i.e., T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Build the tensor in numpy
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # We use a greedy / randomized approach:
    # 1. Start with the full tensor
    # 2. Iteratively find the best rank-1 approximation and subtract
    # 3. Check if the residual is small enough (for exactness, we need rational reconstruction)
    
    # Actually, for exact decomposition with integer/half-integer entries,
    # we need a different approach. Let's use the known structure.
    
    # The matmul tensor for n x m * m x p has rank at most n*m*p (schoolbook).
    # For 3x3x3, the rank is 23 (Strassen). For 3x3x6, we can potentially do better than 54.
    
    # Let's try a randomized approach with rational reconstruction:
    # 1. Use random projections to estimate the rank
    # 2. Use iterative refinement to find a decomposition
    
    # First, let's establish a lower bound by trying ranks
    # The tensor has shape (9, 18, 18)
    
    # Matrix unfolding: flatten to (9, 18*18) = (9, 324)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # The rank of T_mat is an upper bound on the tensor rank
    # Actually, the tensor rank is at least the maximum matrix rank over all unfoldings
    
    # Let's compute the rank of each unfolding
    rank_0 = np.linalg.matrix_rank(T.reshape(n*m, m*p * p*n))
    rank_1 = np.linalg.matrix_rank(T.reshape(m*p, n*m * p*n))
    rank_2 = np.linalg.matrix_rank(T.reshape(p*n, n*m * m*p))
    
    lower_bound = max(rank_0, rank_1, rank_2)
    
    # For the 3x3x6 matmul tensor, let's try to find a decomposition
    # using a randomized iterative approach
    
    # We'll use a greedy rank-1 extraction approach with random initialization
    # and verify exactness using rational arithmetic via fractions
    
    from fractions import Fraction
    
    def verify_exact(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        residual = np.zeros_like(T)
        for c in range(r):
            uc = np.array(U[:, c], dtype=float)
            vc = np.array(V[:, c], dtype=float)
            wc = np.array(W[:, c], dtype=float)
            residual += np.einsum('a,b,c->abc', uc, vc, wc)
        return np.allclose(residual, T, atol=tol)
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # Use random initialization
        random.seed(42)
        np.random.seed(42)
        
        # Initialize with random vectors
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Iterative refinement: alternate between updating U, V, W
        # For tensor decomposition, this is non-convex, but we can try
        
        for iteration in range(100):
            if time.time() > deadline:
                return None
            
            # Update U: fix V and W, solve for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system in U[a,:]
            
            # Build the coefficient matrix
            # For each a, we need to solve: sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a,:,:]
            # Flatten: (V[:,r] * W[:,r]).flatten() is a vector of length m*p*p*n
            # So we have a (m*p*p*n) x r matrix, and T[a,:,:].flatten() is the RHS
            
            # Actually, let's do this more efficiently
            # The coefficient matrix for U is: C_U[a, b*c] = sum_r ... no
            # Let me think again.
            
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed a, we have: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
            # This is a weighted sum of outer products V[:,r] * W[:,r]
            # The coefficients are U[a, r]
            
            # Let's build the matrix M such that M @ u = T[a, :, :].flatten()
            # where u = U[a, :] and M has columns V[:,r] * W[:,r] (flattened)
            
            M = np.zeros((m*p * p*n, r))
            for rr in range(r):
                M[:, rr] = (V[:, rr] * W[:, rr]).flatten()
            
            # For each a, solve M @ U[a,:] = T[a,:,:].flatten()
            for a in range(n*m):
                rhs = T[a, :, :].flatten()
                # Use least squares
                U[a, :], residuals, rank_M, sv = np.linalg.lstsq(M, rhs, rcond=None)
            
            # Similarly update V and W
            M_V = np.zeros((n*m * p*n, r))
            for rr in range(r):
                M_V[:, rr] = (U[:, rr] * W[:, rr]).flatten()
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank_M, sv = np.linalg.lstsq(M_V, rhs, rcond=None)
            
            M_W = np.zeros((n*m * m*p, r))
            for rr in range(r):
                M_W[:, rr] = (U[:, rr] * V[:, rr]).flatten()
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank_M, sv = np.linalg.lstsq(M_W, rhs, rcond=None)
            
            # Check if we've converged
            residual = np.zeros_like(T)
            for c in range(r):
                residual += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            err = np.linalg.norm(residual - T) / np.linalg.norm(T)
            if err < 1e-10:
                # Now try to convert to rational
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                residual2 = np.zeros_like(T)
                for c in range(r):
                    residual2 += np.einsum('a,b,c->abc', U_r[:, c], V_r[:, c], W_r[:, c])
                if np.allclose(residual2, T, atol=1e-8):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None
    
    # Try increasing ranks starting from lower bound
    # For 3x3x6, schoolbook is 54. Let's try from lower_bound up to 54
    best_result = None
    
    # Try a few random restarts for each rank
    for r in range(lower_bound, n*m*p + 1):
        if time.time() > deadline:
            break
        
        found = try_rank(r)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            break
        
        # Try a couple more random restarts for promising ranks
        if r <= 50:
            for restart in range(3):
                if time.time() > deadline:
                    break
                random.seed(42 + restart)
                np.random.seed(42 + restart)
                found = try_rank(r)
                if found is not None:
                    U, V, W = found
                    best_result = (U, V, W)
                    break
            if best_result is not None:
                break
    
    # Fallback to schoolbook if nothing found
    if best_result is None or time.time() > deadline:
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
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
