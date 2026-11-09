"""Rewritten program: attempt a rank-23 exact tensor decomposition for 3x3 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if the term A[i, j'] * B[j', k] contributes to C[k, i]
    # More precisely: T is the 3-way tensor where T[a, b, c] = 1 iff
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # for some (i, j, k) in [0,n) x [0,m) x [0,p)
    # i.e., the "middle" index j is shared between A and B.
    
    # Actually, let's be more careful. The matrix multiplication tensor T_{ijk} for
    # multiplying A (n x m) by B (m x p) to get C (n x p) is:
    # T[i, j, k] = 1 if j is the shared index, i indexes row of A and col of C,
    # k indexes row of B and col of C.
    # In the flattened form:
    #   U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    #   V has shape (m*p, rank), indexed by (j, k) -> j*p + k  
    #   W has shape (p*n, rank), indexed by (k, i) -> k*n + i
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    
    # We want to find U, V, W such that:
    # T = sum_l U[:,l] * V[:,l] * W[:,l]  (outer product sum)
    # i.e., T[a,b,c] = sum_l U[a,l] * V[b,l] * W[c,l]
    
    # Rank 27 schoolbook decomposition:
    rank_schoolbook = n * m * p  # 27
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.zeros_like(T)
        for l in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank(r, max_iter=50000, seed=42):
        """Try to find a rank-r decomposition via iterative refinement."""
        np.random.seed(seed)
        
        # Initialize with random matrices
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Alternating least squares
            # For each l, we have outer product U[:,l] * V[:,l] * W[:,l]
            # This is a sum of rank-1 tensors.
            
            # Update U: for each a, U[a, l] = <T[a, :, :] - sum_{l' != l} U[a,l'] * V[:,l'] * W[:,l'], V[:,l] * W[:,l]> / ||V[:,l] * W[:,l]||^2
            # This is expensive. Let's use a simpler approach.
            
            # Actually, for tensor decomposition, ALS is tricky. Let's use a different approach.
            
            # Compute current reconstruction
            T_recon = np.zeros_like(T)
            for l in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
            
            err = np.linalg.norm(T - T_recon)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-9:
                break
            
            # ALS update:
            # For each l, treat the other (r-1) terms as noise, and solve for U[:,l], V[:,l], W[:,l]
            # This is a rank-1 approximation problem.
            
            # Simpler: use gradient descent on the non-convex objective
            # F = ||T - sum_l U[:,l] V[:,l] W[:,l]||^2
            
            # Gradient with respect to U[a, l]:
            # dF/dU[a,l] = -2 * sum_{b,c} (T[a,b,c] - T_recon[a,b,c]) * V[b,l] * W[c,l]
            
            lr = 0.01
            for l in range(r):
                # Gradient for U[:, l]
                G_U = np.zeros(n * m)
                for b in range(m * p):
                    for c in range(p * n):
                        G_U += (T_recon[:, b, c] - T[:, b, c]) * V[b, l] * W[c, l]
                U[:, l] -= lr * G_U
                
                # Gradient for V[:, l]
                G_V = np.zeros(m * p)
                for a in range(n * m):
                    for c in range(p * n):
                        G_V += (T_recon[a, :, c] - T[a, :, c]) * U[a, l] * W[c, l]
                V[:, l] -= lr * G_V
                
                # Gradient for W[:, l]
                G_W = np.zeros(p * n)
                for a in range(n * m):
                    for b in range(m * p):
                        G_W += (T_recon[a, b, :] - T[a, b, :]) * U[a, l] * V[b, l]
                W[:, l] -= lr * G_W
            
            # Recompute T_recon
            T_recon = np.zeros_like(T)
            for l in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
        
        return best_U, best_V, best_W, best_err
    
    # Try rank 23
    if time.time() < deadline - 20.0:
        U23, V23, W23, err23 = try_rank(23, max_iter=10000, seed=123)
        if check_decomposition(U23, V23, W23, tol=1e-8):
            return {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}
    
    # Try rank 24
    if time.time() < deadline - 15.0:
        U24, V24, W24, err24 = try_rank(24, max_iter=10000, seed=456)
        if check_decomposition(U24, V24, W24, tol=1e-8):
            return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
    
    # Try rank 25
    if time.time() < deadline - 10.0:
        U25, V25, W25, err25 = try_rank(25, max_iter=10000, seed=789)
        if check_decomposition(U25, V25, W25, tol=1e-8):
            return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # Try rank 26
    if time.time() < deadline - 5.0:
        U26, V26, W26, err26 = try_rank(26, max_iter=10000, seed=111)
        if check_decomposition(U26, V26, W26, tol=1e-8):
            return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    # Fallback to schoolbook rank 27
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
