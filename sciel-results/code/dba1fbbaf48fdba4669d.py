import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    # The naive decomposition has rank n*m*p = 24
    # We want to find a lower rank decomposition.
    
    # Strategy: Use the additive structure of the output dimension.
    # The 2x3x4 tensor can be viewed as two 1x3x4 tensors (for i=0 and i=1).
    # Each 1x3x4 tensor has rank at most 3*4 = 12, but we can do better.
    
    # For 1x3x4 matrix multiplication (row vector 1x3 times 3x4 matrix),
    # the output is 1x4. This is equivalent to 4 independent 1x3 dot products.
    # The naive rank is 3*4 = 12.
    
    # However, we can use the structure more cleverly.
    # For the 1x3x4 case, we can use a rank-8 or rank-9 decomposition.
    
    # Let's try a systematic approach:
    # 1. First, try to find a decomposition with rank < 24.
    # 2. Use greedy selection of rank-1 tensors.
    
    # Generate candidate rank-1 tensors with small integer/half-integer entries.
    # To keep the search manageable, we'll use entries in {-1, 0, 1}.
    
    # But 3^6 * 3^12 * 3^8 = 3^26 is too large.
    
    # Instead, let's use a smarter approach based on the structure of T.
    
    # For the 2x3x4 case, we can exploit the fact that:
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to find a decomposition with rank r by solving a system of equations.
    
    # For a given rank r, we need to find U (6 x r), V (12 x r), W (8 x r)
    # such that T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]
    
    # This is a nonlinear system. Let's try a different approach.
    
    # Approach: Use the fact that the tensor has a specific structure.
    # We can group the equations by the value of (i,j,k).
    
    # For each (i,j,k), we have:
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # And for all other (a,b,c), T[a,b,c] = 0.
    
    # Let's try rank r = 12, 13, ..., 23 and see if we can find a solution.
    
    # For a fixed rank r, we can use the following approach:
    # 1. Fix U and V.
    # 2. For each r, we have a tensor u_r ⊗ v_r.
    # 3. We need to find W such that the sum of u_r ⊗ v_r ⊗ w_r equals T.
    
    # This is equivalent to: for each (a,b), the vector (T[a,b,c])_c must be in the span of
    # {sum_r U[a,r] * V[b,r] * w_r}
    
    # Actually, let's think of it as: for each r, the tensor u_r ⊗ v_r ⊗ w_r contributes
    # to the (a,b) slice as U[a,r] * V[b,r] * w_r.
    
    # So for each (a,b), we have:
    # T[a,b,:] = sum_r U[a,r] * V[b,r] * w_r
    
    # This is a linear system in the unknowns w_r (for each r, w_r is a vector of length p*n = 8).
    
    # Let's define: for each (a,b), the equation is:
    # T[a,b,:] = sum_r (U[a,r] * V[b,r]) * w_r
    
    # So if we fix U and V, we can solve for W by solving a linear system for each (a,b).
    
    # But we also need to satisfy the constraint that T[a,b,c] = 0 for most (a,b,c).
    
    # Let's try a different approach: use the known result that the rank of 2x3 matrix
    # multiplication is 7, and extend this to 2x3x4.
    
    # For 2x3x4, we can think of it as: for each of the 4 output columns, we have a 2x3
    # matrix multiplication. But that would give rank 4*7 = 28, which is worse than 24.
    
    # Alternatively, we can use the structure more cleverly.
    
    # Let's try the following approach:
    # 1. Start with the naive rank-24 decomposition.
    # 2. Try to merge some of the rank-1 tensors to reduce the rank.
    
    # Two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 can be merged if they have the same
    # support pattern in a certain sense.
    
    # Actually, let's try a more systematic approach: use the fact that the tensor has
    # a specific structure and try to find a low-rank decomposition using numerical methods.
    
    # Approach: Use alternating least squares (ALS) or a similar method to minimize
    # ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F^2 subject to the constraint that the solution is exact.
    
    # But ALS doesn't guarantee an exact solution.
    
    # Let's try a different approach: use the fact that the tensor is sparse and has a
    # specific structure.
    
    # For the 2x3x4 case, the non-zero entries of T are at positions (i*m+j, j*p+k, k*n+i)
    # for i in {0,1}, j in {0,1,2}, k in {0,1,2,3}.
    
    # There are 2*3*4 = 12 non-zero entries.
    
    # Let's try to find a decomposition with rank r where r < 24.
    
    # For a given rank r, we can use the following approach:
    # 1. Generate random U, V, W with entries in {-1, 0, 1}.
    # 2. Check if the resulting tensor matches T.
    # 3. If not, try again.
    
    # But this is unlikely to work for r < 24.
    
    # Let's try a more structured approach.
    
    # For the 1x3x4 case, the tensor has non-zero entries at (j, j*4+k, k) for j in {0,1,2}, k in {0,1,2,3}.
    # There are 12 non-zero entries.
    
    # The naive rank is 12. Can we do better?
    
    # For 1x3x4, the rank is at most 3*4 = 12, but it might be less.
    
    # Let's try to find a rank-8 or rank-9 decomposition for 1x3x4.
    
    # Actually, let's just try the naive decomposition for now and see if we can improve it.
    
    # For the 2x3x4 case, the naive rank is 24.
    
    # Let's try to find a decomposition with rank < 24 using a greedy approach.
    
    # Generate candidate rank-1 tensors with small integer entries.
    # To keep the search manageable, we'll use a limited set of candidates.
    
    # For U vectors (length 6), use entries in {-1, 0, 1} with at most 2 non-zero entries.
    # For V vectors (length 12), use entries in {-1, 0, 1} with at most 2 non-zero entries.
    # For W vectors (length 8), use entries in {-1, 0, 1} with at most 2 non-zero entries.
    
    # This gives a manageable number of candidates.
    
    # But even this might be too large.
    
    # Let's try a different approach: use the known result that the rank of 2x3 matrix
    # multiplication is 7, and use this to construct a decomposition for 2x3x4.
    
    # For 2x3x4, we can write:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use the rank-7 decomposition for 2x3 matrix multiplication to compute
    # A[i,j] * B[j,k] for each k, but that would give rank 7*4 = 28.
    
    # Alternatively, we can use the structure of the 2x3x4 tensor more cleverly.
    
    # Let's try the following approach:
    # 1. Use the rank-7 decomposition for 2x3 matrix multiplication.
    # 2. Extend this to 2x3x4 by using the same 7 rank-1 tensors for each of the 4 output columns.
    # But that would give rank 7*4 = 28, which is worse than 24.
    
    # Let's try a different approach: use the fact that the 2x3x4 tensor can be written as
    # a sum of two 1x3x4 tensors.
    
    # For i=0: T[0*3+j, j*4+k, k*2+0] = 1 for j in {0,1,2}, k in {0,1,2,3}
    # For i=1: T[1*3+j, j*4+k, k*2+1] = 1 for j in {0,1,2}, k in {0,1,2,3}
    
    # Each 1x3x4 tensor has 12 non-zero entries.
    
    # Let's try to find a low-rank decomposition for each 1x3x4 tensor.
    
    # For 1x3x4, the naive rank is 12. Can we do better?
    
    # Let's try rank 8, 9, 10, 11 for each 1x3x4 tensor.
    
    # If we can find a rank-9 decomposition for each, the total rank would be 18.
    
    # Let's try to find a rank-9 decomposition for 1x3x4.
    
    # For 1x3x4, the tensor T1 has non-zero entries at (j, j*4+k, k) for j in {0,1,2}, k in {0,1,2,3}.
    
    # We need to find U1 (3 x 9), V1 (12 x 9), W1 (4 x 9) such that
    # T1[a,b,c] = sum_r U1[a,r] * V1[b,r] * W1[c,r]
    
    # This is a nonlinear system. Let's try to solve it using numerical methods.
    
    # Approach: Use alternating least squares (ALS).
    
    # Initialize U1, V1, W1 randomly.
    # Iterate:
    # 1. Fix V1 and W1, solve for U1 using least squares.
    # 2. Fix U1 and W1, solve for V1 using least squares.
    # 3. Fix U1 and V1, solve for W1 using least squares.
    
    # Repeat until convergence.
    
    # Then check if the resulting tensor matches T1 exactly.
    
    # Let's implement this.
    
    def als_decompose(T, shape_u, shape_v, shape_w, rank, max_iter=1000, tol=1e-10):
        """
        Use alternating least squares to find a rank-`rank` decomposition of tensor T.
        T has shape (shape_u, shape_v, shape_w).
        Returns (U, V, W) where U has shape (shape_u, rank), etc.
        """
        np.random.seed(42)
        U = np.random.randn(shape_u, rank)
        V = np.random.randn(shape_v, rank)
        W = np.random.randn(shape_w, rank)
        
        for iteration in range(max_iter):
            # Fix V and W, solve for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a, T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So T[a,:,:] = U[a,:] @ M where M has columns V[:,r] ⊗ W[:,r]
            
            M = np.zeros((shape_v * shape_w, rank))
            for r in range(rank):
                M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # For each a, solve: T[a,:,:].flatten() = M @ U[a,:]
            U_new = np.zeros((shape_u, rank))
            for a in range(shape_u):
                rhs = T[a, :, :].flatten()
                U_new[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Fix U and W, solve for V
            M = np.zeros((shape_u * shape_w, rank))
            for r in range(rank):
                M[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            V_new = np.zeros((shape_v, rank))
            for b in range(shape_v):
                # T[:,b,:] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
                # T[:,b,:].reshape(shape_u, -1) = ... 
                # Actually, T[:,b,:] has shape (shape_u, shape_w)
                # T[:,b,:] = sum_r V[b,r] * np.outer(U[:,r], W[:,r])
                # So T[:,b,:].flatten() = M @ V[b,:]
                rhs = T[:, b, :].flatten()
                V_new[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Fix U and V, solve for W
            M = np.zeros((shape_u * shape_v, rank))
            for r in range(rank):
                M[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            W_new = np.zeros((shape_w, rank))
            for c in range(shape_w):
                rhs = T[:, :, c].flatten()
                W_new[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Check convergence
            diff_u = np.linalg.norm(U_new - U)
            diff_v = np.linalg.norm(V_new - V)
            diff_w = np.linalg.norm(W_new - W)
            
            U, V, W = U_new, V_new, W_new
            
            if diff_u < tol and diff_v < tol and diff_w < tol:
                break
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return U, V, W
    
    # Try to find a rank-9 decomposition for the 1x3x4 tensor.
    # For i=0: T1[a,b,c] where a in {0,1,2}, b in {0,...,11}, c in {0,1,2,3}
    # T1[j, j*4+k, k] = 1
    
    T1 = np.zeros((3, 12, 4), dtype=np.float64)
    for j in range(3):
        for k in range(4):
            T1[j, j*4+k, k] = 1.0
    
    # Try rank 9
    U1, V1, W1 = als_decompose(T1, 3, 12, 4, 9, max_iter=500)
    
    # Check if the decomposition is exact
    T1_reconstructed = np.zeros((3, 12, 4), dtype=np.float64)
    for r in range(9):
        T1_reconstructed += np.outer(np.outer(U1[:, r], V1[:, r]), W1[:, r]).reshape(3, 12, 4)
    
    error1 = np.linalg.norm(T1 - T1_reconstructed)
    
    # For i=1: T2[a,b,c] where a in {0,1,2}, b in {0,...,11}, c in {0,1,2,3}
    # T2[j, j*4+k, k] = 1 (same structure, but the indices in the full tensor are different)
    
    T2 = np.zeros((3, 12, 4), dtype=np.float64)
    for j in range(3):
        for k in range(4):
            T2[j, j*4+k, k] = 1.0
    
    # The 1x3x4 tensors for i=0 and i=1 have the same structure, so we can use the same decomposition.
    # But the indices in the full tensor are different.
    
    # For i=0: U indices are 0,1,2; W indices are 0,2,4,6
    # For i=1: U indices are 3,4,5; W indices are 1,3,5,7
    
    # So we can use the same U1, V1, W1 for both, but with different index mappings.
    
    # Let's construct the full decomposition.
    rank_total = 18
    U = np.zeros((6, rank_total), dtype=np.float64)
    V = np.zeros((12, rank_total), dtype=np.float64)
    W = np.zeros((8, rank_total), dtype=np.float64)
    
    # For i=0 (first 9 rank-1 tensors)
    for r in range(9):
        for j in range(3):
            U[j, r] = U1[j, r]
        for b in range(12):
            V[b, r] = V1[b, r]
        for k in range(4):
            W[2*k, r] = W1[k, r]  # W indices for i=0 are 0,2,4,6
    
    # For i=1 (next 9 rank-1 tensors)
    for r in range(9):
        for j in range(3):
            U[3+j, 9+r] = U1[j, r]
        for b in range(12):
            V[b, 9+r] = V1[b, r]
        for k in range(4):
            W[2*k+1, 9+r] = W1[k, r]  # W indices for i=1 are 1,3,5,7
    
    # Check if the full decomposition is exact
    T_reconstructed = np.zeros((6, 12, 8), dtype=np.float64)
    for r in range(rank_total):
        T_reconstructed += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(6, 12, 8)
    
    error = np.linalg.norm(T - T_reconstructed)
    
    if error < 1e-6:
        # Found an exact decomposition with rank 18
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If the rank-9 decomposition didn't work, try the naive rank-24 decomposition.
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
