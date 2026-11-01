"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    We start with the schoolbook rank (42) and attempt to reduce it using numerical methods and randomized approaches.
    If we cannot find a valid lower-rank decomposition, we return the schoolbook decomposition.
    
    The score is the rank, and we aim to minimize it.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Function to create the schoolbook decomposition
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Function to check if a decomposition is valid using vectorized computation
    def check_decomposition(decomp, tol=1e-8):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        # Check shapes
        if U.shape[0] != n * m:
            return False
        if V.shape[0] != m * p:
            return False
        if W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Vectorized check: compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and verify it equals 1 for all valid (i,j,k)
        
        # Reshape for efficient computation
        U_mat = U.reshape(n, m, rank)  # U[i,j,r]
        V_mat = V.reshape(m, p, rank)  # V[j,k,r]
        W_mat = W.reshape(p, n, rank)  # W[k,i,r]
        
        # Compute the tensor product sum
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U_mat[i, j, :], V_mat[j, k, :])
                    actual = np.dot(actual, W_mat[k, i, :])
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    # Function to try to reduce rank using alternating least squares on the tensor
    def try_reduce_rank_alt(target_rank, max_iterations=2000, tol=1e-10, seed=42):
        np.random.seed(seed)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Alternating least squares: fix two factors, optimize the third
            # Update U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:])^T = 1 for all k
            # So: U[i*m+j,:] * A = 1^T where A[k,r] = sum... 
            # Actually: for fixed i,j: sum_r U[i,j,r] * (sum_k V[j,k,r] W[k,i,r] delta(k,k')) = 1
            # Let's define B[j,i,k',r] = V[j,k',r] * W[k',i,r]
            # Then: sum_r U[i,j,r] * B[j,i,k',r] = 1 for all k'
            # So: U[i,j,:] * B[j,i,:,:]^T = 1^T
            # U[i,j,:] = 1^T * (B[j,i,:,:] B[j,i,:,:]^T)^{-1} if invertible
            
            for i in range(n):
                for j in range(m):
                    # B[j,i,k',r] = V[j*p+k',r] * W[k'*n+i,r]
                    B = V[j*p:(j+1)*p, :] * W[np.arange(p)*n+i, :].T
                    # B shape: (p, target_rank)
                    # We want: U[i*m+j,:] @ B.T = ones(p)
                    # U[i*m+j,:] = ones(p) @ pinv(B.T)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            # Let C[i,j,k,r] = U[i*m+j,r] * W[k*n+i,r]
            # Then: sum_r V[j*p+k,r] * C[i,j,k,r] = 1 for all i
            # V[j*p+k,:] = ones(n) @ pinv(C[:,j,k,:].T)
            for j in range(m):
                for k in range(p):
                    # C[i,j,k,r] = U[i*m+j,r] * W[k*n+i,r]
                    C = U[np.arange(n)*m+j, :] * W[k*n + np.arange(n), :].T
                    # C shape: (n, target_rank)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(C.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            # Let D[j,k,i,r] = U[i*m+j,r] * V[j*p+k,r]
            # Then: sum_r W[k*n+i,r] * D[j,k,i,r] = 1 for all j
            # W[k*n+i,:] = ones(m) @ pinv(D[:,k,i,:].T)
            for k in range(p):
                for i in range(n):
                    # D[j,k,i,r] = U[i*m+j,r] * V[j*p+k,r]
                    D = U[i*m + np.arange(m), :] * V[np.arange(m)*p+k, :].T
                    # D shape: (m, target_rank)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(D.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            error = 0.0
            U_mat = U.reshape(n, m, target_rank)
            V_mat = V.reshape(m, p, target_rank)
            W_mat = W.reshape(p, n, target_rank)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.dot(np.dot(U_mat[i, j, :], V_mat[j, k, :]), W_mat[k, i, :])
                        error += (actual - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_error < tol:
                break
            
            # Occasional random restart if stuck
            if it > 0 and it % 500 == 0 and best_error > 1e-4:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
        
        decomp = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        if check_decomposition(decomp, tol=1e-8):
            return decomp
        return None
    
    # Function to try random restarts with different seeds
    def try_reduce_rank_multi_seed(target_rank, n_seeds=10, max_iterations=1000, tol=1e-10):
        for seed in range(n_seeds):
            if time.time() > deadline:
                break
            result = try_reduce_rank_alt(target_rank, max_iterations=max_iterations, tol=tol, seed=seed)
            if result is not None:
                return result
        return None
    
    # Start with schoolbook
    best_decomp = schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    # Try to reduce rank starting from lower targets
    # Known lower bounds: for 2x3x7, the rank is at least 18 (from Strassen-type bounds)
    # Try from 18 upward to find the minimal rank
    for target_rank in range(18, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        
        result = try_reduce_rank_multi_seed(target_rank, n_seeds=5, max_iterations=500, tol=1e-10)
        if result is not None:
            best_decomp = result
            best_rank = target_rank
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
