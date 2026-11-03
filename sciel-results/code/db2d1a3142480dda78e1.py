"""Matrix multiplication 3x3 x 3x7 rank minimization."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    for idx in range(rank0):
        i = idx // (m*p)
        j = (idx // p) % m
        k = idx % p
        U0[i*m+j, idx] = 1
        V0[j*p+k, idx] = 1
        W0[k*n+i, idx] = 1
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    def verify(U, V, W, tol=1e-9):
        """Check if U @ diag @ V^T @ ... reconstructs T"""
        # Reconstruct: sum_c U[:,c] * V[:,c] * W[:,c] should equal T
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        r = U.shape[1]
        # Reshape for batch computation
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a 3-tensor contraction
        # Use einsum
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i,j,k] = 1.0
        
        return np.max(np.abs(T_recon - T_target)) < tol
    
    def try_reduce(U, V, W, max_rounds=50):
        """Try to reduce rank by finding linearly dependent columns"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Try to find a column that is a linear combination of others
            # Use the structure: if column c can be expressed as linear combo of others
            # in U, V, W simultaneously, we can eliminate it
            
            # Simple approach: check if any column is nearly zero
            for c in range(r):
                if np.max(np.abs(U[:, c])) < 1e-12 and \
                   np.max(np.abs(V[:, c])) < 1e-12 and \
                   np.max(np.abs(W[:, c])) < 1e-12:
                    # Remove this column
                    mask = np.ones(r, dtype=bool)
                    mask[c] = False
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    r -= 1
                    break
            else:
                break
        
        return U, V, W, r
    
    # Try structured rank reduction using known techniques
    # For 3x3 x 3x7, we can exploit the structure
    
    # Method 1: Group by (i,k) pairs - for each (i,k), we have m=3 terms
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For fixed (i,k), this is a vector in j: sum_c (U[i,:,c]*W[k,i,c]) * V[:,k,c]
    
    # Let's try a different approach: use the fact that for fixed i,k,
    # the j-dependence is linear, so we might share factors
    
    # Actually, let's try a direct optimization approach
    # Start from schoolbook and try to merge columns
    
    # Key insight: columns (i1,j1,k1) and (i2,j2,k2) can be merged if
    # the resulting rank-2 tensor can be decomposed into rank 1
    
    # Try pairwise merging
    def try_merge(U, V, W, c1, c2):
        """Try to merge columns c1 and c2 into one column"""
        r = U.shape[1]
        if c1 >= r or c2 >= r:
            return None
        
        # We want to find a single column that represents both
        # This is possible if the 2x2 sub-problem has rank 1
        # For the merge to work, we need:
        # U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] = U'[:,c]*V'[:,c]*W'[:,c]
        # This is generally not possible, but we can try to absorb one into the other
        
        # Alternative: try to express one column as a perturbation
        # and see if we can adjust other columns to compensate
        
        return None
    
    # Better approach: use the known structure of matrix multiplication
    # For 3x3 x 3x7, we can use a block structure
    
    # Let's try a systematic search over small rank reductions
    # Start with rank 63 and try to reduce
    
    # Actually, let's use a more principled approach:
    # The tensor T has a specific structure. Let's try to find a low-rank
    # decomposition using iterative methods.
    
    # Method: Alternating least squares / tensor train approach
    # But for exact decomposition, we need rational/integer entries
    
    # Let's try a greedy approach:
    # 1. Start with schoolbook
    # 2. For each pair of columns, check if they can be combined
    # 3. Use linear algebra to find dependencies
    
    # For 3x3 x 3x7, known results suggest rank around 50-55
    # Let's try to achieve this through structured construction
    
    # Construction idea:
    # For the 3x3 part, use the 3x3x3 tensor (rank 9 for schoolbook, but can be lower)
    # Actually, 3x3x3 matrix mult has rank 23 (Strassen-like)
    # For 3x3 x 3x7, we can think of it as 7 copies of 3x3 x 3x1
    # But that's not quite right...
    
    # Let's try a direct construction:
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We want sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # This is equivalent to finding a low-rank decomposition of the tensor
    
    # Use a randomized initialization + local optimization
    def optimize_rank(U_init, V_init, W_init, target_rank, max_iters=1000):
        """Try to find a rank-target decomposition"""
        r = target_rank
        # Random initialization
        np.random.seed(42)
        U = U_init[:, :r].copy() if U_init.shape[1] >= r else np.random.randn(n*m, r)
        V = V_init[:, :r].copy() if V_init.shape[1] >= r else np.random.randn(m*p, r)
        W = W_init[:, :r].copy() if W_init.shape[1] >= r else np.random.randn(p*n, r)
        
        # Scale to help convergence
        for c in range(r):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U * norm_V * norm_W > 0:
                scale = (norm_U * norm_V * norm_W) ** (1/3)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), minimize ||sum_c U[i,j,c]*V[:,c]*W[:,c] - T[i,j,:]||
            # This is a linear system in U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i,j,c] * V[j*p+k, c] * W[k*n+i, c]
                    # Let A[k,c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then T[i,j,:] = U[i,j,:] @ A
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i,j,:] @ A = T[i,j,:]
                    # This is underdetermined if r > p, use least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Let B[i,c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then T[:,j,k] = B @ V[j*p+k,:]
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Let C[j,c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then T[i,:,k] = C @ W[k*n+i,:]
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        return U, V, W
    
    # Try various target ranks
    for target_rank in range(63, 45, -1):
        if time.time() > deadline:
            break
        
        U_opt, V_opt, W_opt = optimize_rank(U0, V0, W0, target_rank, max_iters=200)
        
        if verify(U_opt, V_opt, W_opt):
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            best_rank = target_rank
            break
    
    # Convert to list format
    U_list = [[float(U_opt[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(V_opt[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(W_opt[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
