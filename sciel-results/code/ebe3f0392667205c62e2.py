"""Matrix multiplication 2x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 2x5x5 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (10, rank)
        - V has shape (m*p, rank) = (25, rank)  
        - W has shape (p*n, rank) = (10, rank)
        
    The decomposition satisfies:
    C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    = sum_j A[i,j] * B[j,k] for all A, B.
    
    This is equivalent to: for each (i,j,k), 
    sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k*n+i',c] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    where the sum is over all (i',j',k') combinations weighted by the product.
    
    More precisely, the tensor T[i,j,k] = delta_{j,j'}*delta_{i,i'}*delta_{k,k'} must be
    decomposed as sum_c U[:,c] outer V[:,c] outer W[:,c] in the appropriate tensor product.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10
    
    # Build the target matrix multiplication tensor M of shape (dim_U, dim_V, dim_W)
    # M[a, b, c] where a = i*m+j, b = j'*p+k', c = k*n+i'
    # The bilinear map requires: sum_c U[a,c]*V[b,c]*W[c_idx,c] = M[a,b,c_idx]
    # where M[a,b,c_idx] = 1 if (i,j) from a, (j',k') from b, (k,i') from c_idx satisfy
    # j=j', i=i', k=k', else 0.
    
    M = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c_idx = k * n + i
                    M[a, b, c_idx] = 1.0
    
    # The schoolbook rank is n*m*p = 50. Let's try to find lower rank.
    # We'll use a randomized approach: try various ranks and use iterative refinement.
    
    deadline = time.time() + 100  # Safety margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_init = np.zeros((dim_U, best_rank))
    V_init = np.zeros((dim_V, best_rank))
    W_init = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m + j, col] = 1.0
                V_init[j*p + k, col] = 1.0
                W_init[k*n + i, col] = 1.0
                col += 1
    
    def residual(U, V, W):
        """Compute the residual ||M - sum_c U[:,c] outer V[:,c] outer W[:,c]||_F"""
        # Build the reconstructed tensor
        recon = np.zeros_like(M)
        for c in range(U.shape[1]):
            recon += np.einsum('a,b,d->abd', U[:,c], V[:,c], W[:,c])
        return np.linalg.norm(M - recon)
    
    def try_rank(r, max_iters=500, lr=0.01, seed=42):
        """Try to find a decomposition of rank r."""
        if r <= 0:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize with random Gaussian matrices
        U = rng.randn(dim_U, r) * 0.1
        V = rng.randn(dim_V, r) * 0.1
        W = rng.randn(dim_W, r) * 0.1
        
        # Use a gradient-free approach: iterative multiplicative updates
        # or use a simpler approach with power iterations on the tensor.
        
        # Let's try an approach based on alternating least squares or
        # a simpler heuristic: try small ranks first with structured initialization.
        
        # Actually, let's use a more principled approach.
        # For matrix multiplication C = AB, there are known lower-rank decompositions
        # for certain dimensions. For 2x5x5, the known rank is not well-established
        # but it's at most 50 and at least max(n*m, m*p, p*n) = 25.
        
        # Let me try a different approach: use the fact that we can combine
        # multiple (i,j,k) triples into single rank-1 terms if the patterns align.
        
        # Try structured initialization based on block structure
        if r < n * m * p:
            # Try to group indices
            # For each j, we have n*p = 2*5 = 10 triples (i,k)
            # Maybe we can do better than one per (i,j,k)
            
            pass
        
        # Fallback: just return None and use schoolbook
        return None
    
    # Given the complexity and time constraints, let's try a few specific constructions.
    
    # Construction 1: Schoolbook (rank 50)
    U_sb = np.zeros((dim_U, 50))
    V_sb = np.zeros((dim_V, 50))
    W_sb = np.zeros((dim_W, 50))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    if time.time() > deadline - 20:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find if there's a known lower rank.
    # The minimum rank for matrix multiplication of an (n x m) matrix by (m x p) matrix
    # is at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
    # It's also at most n*m*p = 50.
    
    # Let me try a simple heuristic search for rank between 25 and 50.
    # Use a randomized iterative method.
    
    def gradient_step(U, V, W, M, lr):
        """Perform one gradient step to minimize ||M - sum_c U[:,c]V[:,c]W[:,c]||^2."""
        # Compute current reconstruction
        recon = np.zeros_like(M)
        for c in range(U.shape[1]):
            recon += np.einsum('a,b,d->abd', U[:,c], V[:,c], W[:,c])
        
        # Gradient w.r.t. U[a,c]: sum_{b,d} (M[a,b,d] - recon[a,b,d]) * V[b,c] * W[d,c]
        # = einsum('abd,bc,dc->ac', M - recon, V, W)
        grad_U = np.einsum('abd,bc,dc->ac', M - recon, V, W)
        grad_V = np.einsum('abd,ac,dc->bc', M - recon, U, W)
        grad_W = np.einsum('abd,ac,bc->dc', M - recon, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        return U, V, W
    
    # Try a few ranks with gradient descent
    for target_rank in [25, 30, 35, 40, 45]:
        if time.time() > deadline - 30:
            break
            
        for seed in range(3):
            if time.time() > deadline - 25:
                    break
            rng = np.random.RandomState(seed)
            U = rng.randn(dim_U, target_rank) * 0.1
            V = rng.randn(dim_V, target_rank) * 0.1
            W = rng.randn(dim_W, target_rank) * 0.1
            
            # Run gradient descent with adaptive learning rate
            lr = 0.001
            best_res = float('inf')
            best_U_copy = U.copy()
            best_V_copy = V.copy()
            best_W_copy = W.copy()
            
            for it in range(100):
                if time.time() > deadline - 20:
                    break
                    
                U, V, W = gradient_step(U, V, W, M, lr)
                res = residual(U, V, W)
                
                if res < best_res:
                    best_res = res
                    best_U_copy = U.copy()
                    best_V_copy = V.copy()
                    best_W_copy = W.copy()
                
                # Adaptive learning rate
                if res < best_res * 0.9:
                    lr = min(lr * 1.5, 0.01)
                else:
                    lr = lr * 0.95
                
                if best_res < 1e-10:
                    break
            
            if best_res < 1e-10:
                # Found a valid decomposition!
                # Clean up to exact integers if possible
                U_clean = np.round(best_U_copy, 8)
                V_clean = np.round(best_V_copy, 8)
                W_clean = np.round(best_W_copy, 8)
                
                # Verify
                recon_final = np.zeros_like(M)
                for c in range(target_rank):
                    recon_final += np.einsum('a,b,d->abd', U_clean[:,c], V_clean[:,c], W_clean[:,c])
                final_res = np.linalg.norm(M - recon_final)
                
                if final_res < 1e-8:
                    return {
                        "U": U_clean.tolist(),
                        "V": V_clean.tolist(),
                        "W": W_clean.tolist()
                    }
    
    # If nothing worked, return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
