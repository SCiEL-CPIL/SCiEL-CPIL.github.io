"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 36
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                U_school[alpha, col] = 1
                V_school[beta, col] = 1
                W_school[gamma, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Try to find lower rank decompositions
    # Strategy: use structured initialization inspired by DFT/circulant structure
    # For small matrices, try rank from 20 to 35
    
    best_result = to_lists(U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # The tensor T[i,j,k] = 1 if k*n+i is the output index, 
    # where we multiply A[i,j] * B[j,k] -> C[i,k]
    # T[alpha, beta, gamma] where alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # T[alpha,beta,gamma] = 1 iff there exist i,j,k with alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank(rank, seed=0, max_iter=5000):
        """Try to find a decomposition of given rank."""
        if rank >= best_rank:
            return None
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize initial columns
        for r in range(rank):
            norm = np.linalg.norm(U[:,r])
            if norm > 1e-10:
                U[:,r] /= norm
                V[:,r] *= norm
                W[:,r] *= norm
        
        # Alternating least squares
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each column r, solve for U[:,r] given V[:,r], W[:,r]
            # T = sum_r U_r outer V_r outer W_r
            # For fixed r, we want U[:,r] such that U[:,r] outer (V[:,r] outer W[:,r]) approximates residual
            # This is a linear system
            
            # Build linear system for U
            # T = sum_r U[:,r] * (V[:,r] * W[:,r])^T flattened
            # Let's use a different approach: gradient descent on the residual
            
            # Compute current reconstruction
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            residual = T - T_recon
            
            # Check convergence
            if np.max(np.abs(residual)) < 1e-8:
                if verify_decomposition(U, V, W, tol=1e-6):
                    return U, V, W
                break
            
            # Gradient-based update
            # d/dU[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            for r in range(rank):
                # Gradient for U[:,r]
                grad_U = np.einsum('abc,b,c->a', residual, V[:,r], W[:,r])
                U[:,r] -= 0.01 * grad_U
                
                grad_V = np.einsum('abc,a,c->b', residual, U[:,r], W[:,r])
                V[:,r] -= 0.01 * grad_V
                
                grad_W = np.einsum('abc,a,b->c', residual, U[:,r], V[:,r])
                W[:,r] -= 0.01 * grad_W
            
            # Renormalize to prevent scale drift
            for r in range(rank):
                norms = [np.linalg.norm(U[:,r]), np.linalg.norm(V[:,r]), np.linalg.norm(W[:,r])]
                geom_mean = (norms[0] * norms[1] * norms[2]) ** (1.0/3.0)
                if geom_mean > 1e-10:
                    U[:,r] /= norms[0]
                    V[:,r] /= norms[1]
                    W[:,r] /= norms[2]
                    scale = geom_mean ** 3
                    U[:,r] *= scale ** (1.0/3.0)
                    V[:,r] *= scale ** (1.0/3.0)
                    W[:,r] *= scale ** (1.0/3.0)
        
        # Final check
        if verify_decomposition(U, V, W, tol=1e-6):
            return U, V, W
        return None
    
    # Try various ranks with multiple seeds
    for rank in range(20, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = try_rank(rank, seed=seed, max_iter=2000)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = to_lists(U, V, W)
                break  # Found this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
