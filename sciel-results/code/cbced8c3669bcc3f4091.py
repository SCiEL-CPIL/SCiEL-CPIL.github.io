"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # Precompute the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We'll work with the flattened index mapping.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The tensor is T[a, b, c] where:
    # a = i*m + j  (i in 0..n-1, j in 0..m-1)
    # b = j*p + k  (j in 0..m-1, k in 0..p-1)
    # c = k*n + i  (k in 0..p-1, i in 0..n-1)
    # T[a, b, c] = 1 if the indices are consistent (same i, j, k), else 0.
    
    # We can represent the tensor as a 3D array of shape (n*m, m*p, p*n)
    # But that's 15*30*18 = 8100 elements, which is fine.
    
    # Let's build the tensor explicitly.
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # This is a tensor decomposition problem. We can use various approaches:
    # 1. Start with schoolbook (rank = n*m*p = 90) and try to reduce.
    # 2. Use Strassen-like block decompositions.
    # 3. Use random initialization and optimization.
    
    # Let's first try a structured approach using block Strassen.
    # The matrix multiplication is (n x m) * (m x p) = (n x p).
    # Here n=3, m=5, p=6.
    
    # One idea: partition the matrices into blocks and use lower-rank block multiplications.
    # For example, partition the 3x5 matrix A into blocks and the 5x6 matrix B into blocks.
    
    # Let's try a different approach: use the fact that the tensor is sparse and has structure.
    # We can try to find a decomposition using iterative methods.
    
    # Approach: Start with a random initialization of U, V, W with some rank r,
    # and use gradient descent or similar to minimize the error.
    
    # But the entries must be integer or half-integer. So we need to round at the end.
    
    # Let's try a more structured approach first.
    
    # Block decomposition idea:
    # Partition A (3x5) into blocks of size 1x1, 1x2, 2x1, etc.
    # Partition B (5x6) into corresponding blocks.
    
    # Actually, let's try a simpler approach: use the known lower bounds and known decompositions.
    # The rank of matrix multiplication tensor for (n x m) * (m x p) is at least max(n*m, m*p, n*p) = max(15, 30, 18) = 30.
    # The schoolbook rank is 90.
    
    # Let's try to use a rank-7 Strassen decomposition on 2x2x2 sub-blocks.
    # But our dimensions are 3, 5, 6, which don't align perfectly with 2x2x2 blocks.
    
    # Alternative: use a direct search for a low-rank decomposition.
    # We can use the "tensor rank" algorithms, but they are NP-hard in general.
    
    # For a practical approach, let's try:
    # 1. Start with schoolbook (rank 90).
    # 2. Try to reduce rank by identifying redundant terms.
    # 3. Use random restarts with optimization.
    
    # Let's implement a simple approach: random initialization + optimization.
    
    def verify(U, V, W, T):
        """Verify that U, V, W decompose T."""
        r = U.shape[1]
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Compute the decomposition.
        # This is O(n*m * m*p * p*n * r) which is 15*30*18*r = 8100*r.
        # For r=90, that's 729000 operations, which is fine.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    def verify_exact(U, V, W, T):
        """Verify exact decomposition with integer/half-integer entries."""
        r = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Try random initialization with various ranks.
    best_result = None
    best_rank = n * m * p  # start with schoolbook
    
    # First, create the schoolbook decomposition as a fallback.
    U_school = np.zeros((n*m, best_rank), dtype=np.int64)
    V_school = np.zeros((m*p, best_rank), dtype=np.int64)
    W_school = np.zeros((p*n, best_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Verify schoolbook.
    if verify_exact(U_school, V_school, W_school, T):
        best_result = (U_school, V_school, W_school)
    
    # Now try to find a better decomposition.
    # We'll try ranks from 30 to 90.
    
    for rank in range(30, 91):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations for this rank.
        n_attempts = 5 if rank < 60 else 3
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integer/half-integer values.
            U = np.random.randint(-2, 3, (n*m, rank)) * 0.5
            V = np.random.randint(-2, 3, (m*p, rank)) * 0.5
            W = np.random.randint(-2, 3, (p*n, rank)) * 0.5
            
            # Check if this random initialization happens to work (unlikely but possible).
            if verify_exact(U, V, W, T):
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
                break
            
            # Try a simple optimization: iteratively update one factor at a time.
            # This is similar to alternating least squares for tensor decomposition.
            
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, U[a, :] should satisfy
                # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c] for all b, c.
                # This is a linear system in U[a, :].
                # Let M[a, b, c] = V[b, :] * W[c, :] (outer product, shape (b, c, r)).
                # Then sum_r U[a, r] * M[a, b, c, r] = T[a, b, c].
                # Reshape to a 2D system: for fixed a, we have variables U[a, 0..r-1].
                # The equation is: sum_r U[a, r] * (V[b, r] * W[c, r]) = T[a, b, c]
                # for all b, c. This is a linear system with r unknowns and (m*p * p*n) equations.
                
                # Let's build the matrix A_a of shape (m*p * p*n, r) and vector b_a of shape (m*p * p*n,).
                # A_a[(b, c), r] = V[b, r] * W[c, r]
                # b_a[(b, c)] = T[a, b, c]
                
                A_a = np.einsum('br,cr->bcr', V, W).reshape(m*p * p*n, rank)
                b_a = T[a].reshape(-1)
                
                # Solve the least squares problem.
                try:
                    U_new_a, residuals, rank_ls, sv = np.linalg.lstsq(A_a, b_a, rcond=None)
                    U[a] = U_new_a
                except:
                    pass
                
                # Update V similarly.
                A_b = np.einsum('ar,cr->acr', U, W).reshape(n*m * p*n, rank)
                for b in range(m*p):
                    # Extract T[:, b, :] which has shape (n*m, p*n).
                    T_slice = T[:, b, :].reshape(-1)
                    b_b = T_slice
                    try:
                        V_new_b, residuals, rank_ls, sv = np.linalg.lstsq(A_b, b_b, rcond=None)
                        V[b] = V_new_b
                    except:
                        pass
                
                # Update W similarly.
                A_c = np.einsum('ar,br->abr', U, V).reshape(n*m * m*p, rank)
                for c in range(p*n):
                    T_slice = T[:, :, c].reshape(-1)
                    b_c = T_slice
                    try:
                        W_new_c, residuals, rank_ls, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                        W[c] = W_new_c
                    except:
                        pass
                
                # Round to nearest half-integer.
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Check if we have an exact decomposition.
                if verify_exact(U, V, W, T):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    break
            
            if best_result is not None and best_rank == rank:
                break
    
    if best_result is None:
        best_result = (U_school, V_school, W_school)
    
    U, V, W = best_result
    
    # Convert to lists of lists as required.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
