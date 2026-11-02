"""Rewritten program for matrix multiplication 2x4 * 4x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p  # 2*4*5 = 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 130  # leave 20s margin

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # The tensor T[i,j,k] = 1 for all i,j,k (standard matrix multiplication)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)

    # Try alternating least squares / structured approaches
    # Lower bound considerations:
    # - rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # - Known results suggest rank around 24-28 for 2x4*4x5

    def verify_decomp(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        rank = U.shape[1]
        # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_decomp_fast(U, V, W, tol=1e-6):
        """Faster verification using numpy."""
        rank = U.shape[1]
        # Reshape for batch computation
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Create arrays of shape (n, m, p, rank)
        # U_expanded: U[i*m+j, c] -> shape (n, m, 1, rank)
        U_exp = U.reshape(n, m, 1, rank)
        V_exp = V.reshape(1, m, p, rank)
        W_exp = W.reshape(p, 1, n, rank)
        
        # We need sum over c of U[i,m,0,c] * V[0,m,p,c] * W[p,0,n,c]
        # But indices need to match: U[i*m+j], V[j*p+k], W[k*n+i]
        # So for fixed (i,j,k): U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's build it properly
        # U_idx[i,j,c] = U[i*m+j, c], shape (n,m,rank)
        # V_idx[j,k,c] = V[j*p+k, c], shape (m,p,rank)  
        # W_idx[k,i,c] = W[k*n+i, c], shape (p,n,rank)
        
        # Product at (i,j,k) = sum_c U_idx[i,j,c] * V_idx[j,k,c] * W_idx[k,i,c]
        
        # Use einsum
        # U_idx has shape (n, m, rank)
        # V_idx has shape (m, p, rank)
        # W_idx has shape (p, n, rank)
        
        # result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        
        U_idx = U.reshape(n, m, rank)
        V_idx = V.reshape(m, p, rank)
        W_idx = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True

    def als_decompose(target_rank, max_iters=500, tol=1e-8, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
                
            # Fix V and W, solve for U
            # For each (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system: for fixed (i,j), U[i*m+j,:] * M_ij = 1 where
            # M_ij[k,c] = V[j*p+k,c] * W[k*n+i,c]
            
            for i in range(n):
                for j in range(m):
                    # Build matrix M of shape (p, target_rank)
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M^T * U[i*m+j,:]^T = ones(p)
                    # i.e., U[i*m+j,:] * M^T = ones(p)
                    # So U[i*m+j,:] = ones(p) * pinv(M^T) ... 
                    # Actually: sum_c U[c] * M[k,c] = 1 for all k
                    # M^T @ U_row = ones(p) where U_row has shape (target_rank,)
                    # Wait: sum_c M[k,c] * U[c] = 1 for all k
                    # That's M @ U = ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        U[i * m + j, :] = np.linalg.pinv(M) @ np.ones(p)
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        V[j * p + k, :] = np.linalg.pinv(M) @ np.ones(n)
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        W[k * n + i, :] = np.linalg.pinv(M) @ np.ones(m)
            
            # Check convergence
            if verify_decomp_fast(U, V, W, tol=1e-5):
                return U, V, W, True
        
        # Check if we got a valid decomposition
        if verify_decomp_fast(U, V, W, tol=1e-4):
            return U, V, W, True
        return U, V, W, False

    # Try various ranks
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
            
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            if found:
                break
                
            U, V, W, valid = als_decompose(target_rank, max_iters=300, tol=1e-8, seed=seed)
            if valid:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                # If we found a valid decomposition, try lower ranks
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
