"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses Iterative Hard Thresholding (IHT) to minimize the number of non-zero rank-1 terms,
    leveraging the high sparsity in the structural representation of the MM tensor.
    """
    n, m, p = DIMENSIONS
    
    import numpy as np
    import random
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def validate_decomposition(U, V, W, rank):
        """Check if U*V*W decomposition is valid."""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def iht_decompose(target_tensor, rank, max_iter=500, tol=1e-12):
        """
        Use Iterative Hard Thresholding to find a rank-r decomposition.
        Returns (U, V, W) if found, else None.
        """
        dim1, dim2, dim3 = target_tensor.shape
        
        # Initialize with random factors
        np.random.seed(42)
        U = np.random.randn(dim1, rank) * 0.1
        V = np.random.randn(dim2, rank) * 0.1
        W = np.random.randn(dim3, rank) * 0.1
        
        for iteration in range(max_iter):
            # Compute current reconstruction
            T_current = np.zeros((dim1, dim2, dim3))
            for r in range(rank):
                T_current += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Compute residual
            residual = target_tensor - T_current
            
            # For each rank-1 component, compute the optimal update
            # This is the "projection" step - find the best rank-r approximation
            # We use a simplified approach: for each term, compute the gradient
            
            # Gradient of ||target - sum_r u_r o v_r o w_r||^2 w.r.t. u_r:
            # -2 * sum_{b,c} (target[a,b,c] - T_current[a,b,c]) * v_r[b] * w_r[c]
            
            # Update each factor using gradient information
            U_new = np.zeros_like(U)
            V_new = np.zeros_like(V)
            W_new = np.zeros_like(W)
            
            for r in range(rank):
                # Compute gradients for term r
                # grad_U[a] = -2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                grad_U = -2.0 * np.einsum('abc,b,c->a', residual, V[:, r], W[:, r])
                grad_V = -2.0 * np.einsum('abc,a,c->b', residual, U[:, r], W[:, r])
                grad_W = -2.0 * np.einsum('abc,a,b->c', residual, U[:, r], V[:, r])
                
                # Simple gradient step
                lr = 0.01
                U_new[:, r] = U[:, r] + lr * grad_U
                V_new[:, r] = V[:, r] + lr * grad_V
                W_new[:, r] = W[:, r] + lr * grad_W
            
            U, V, W = U_new, V_new, W_new
            
            # Hard thresholding: keep only the r most significant terms
            # Measure "significance" by the norm of each rank-1 component's contribution
            sigmas = []
            for r in range(rank):
                term = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                sigmas.append(np.linalg.norm(term))
            
            # Keep the top-r terms by significance
            # If all are similar, keep all
            if len(set(np.round(sigmas, 6))) > 1:
                # Sort by significance and keep top rank terms
                # But we need exactly rank terms, so this is a no-op if we already have rank terms
                pass
            
            # Check convergence
            error = np.linalg.norm(residual)
            if error < tol:
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 24
    upper_bound = n * m * p  # 72
    
    # First, try the structured schoolbook solution as a fallback
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try IHT for ranks below schoolbook
    for rank in range(lower_bound, schoolbook_rank):
        # Try multiple random seeds
        found = False
        for seed in range(10):
            np.random.seed(seed)
            result = iht_decompose(T, rank, max_iter=200, tol=1e-10)
            if result is not None:
                U, V, W = result
                if validate_decomposition(U, V, W, rank):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
