import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target rank
    target_rank = 16
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition
    # We use a simple alternating minimization
    # Initialize with random matrices
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        rank = target_rank
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to prevent divergence
        for r in range(rank):
            norm = np.linalg.norm(U[:, r])
            if norm > 1e-6:
                U[:, r] /= norm
                V[:, r] *= norm
            norm = np.linalg.norm(V[:, r])
            if norm > 1e-6:
                V[:, r] /= norm
                W[:, r] *= norm
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Update U
            # T = sum_r U[:,r] V[:,r]^T W[:,r]^T
            # We want to minimize ||T - T_target||
            # T_target is the matrix multiplication tensor
            # T_target[i*m+j, j*p+k, k*n+i] = 1 if i,j,k are valid
            # Actually, the tensor is defined by the multiplication.
            # Let's construct the target tensor T_target of shape (n*m, m*p, p*n)
            # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
            # This is a sparse tensor.
            
            # Instead of constructing the full tensor, we can use the fact that
            # T_target = sum_{i,j,k} e_{i*m+j} e_{j*p+k}^T e_{k*n+i}^T
            # We can compute the error by projecting onto the basis.
            
            # Let's use a simpler approach: 
            # The decomposition is exact if U V^T W^T = T_target
            # We can check this by computing the product and comparing.
            
            # To update U, we fix V and W.
            # The term for U is linear.
            # Let's use a gradient descent step.
            
            # Compute current T
            # T = np.einsum('ar,br,cr->abc', U, V, W)
            # This is too slow for large rank and dimensions?
            # n*m=6, m*p=18, p*n=12. Rank=16.
            # T size 6x18x12. Small.
            
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            
            # T_target
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m + j, j*p + k, k*n + i] = 1
            
            error = np.linalg.norm(T_current - T_target)
            
            if error < 1e-6:
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
            
            # Gradient for U
            # dL/dU = (T_current - T_target) contracted with V and W
            # dL/dU[a, r] = sum_{b,c} (T_current[a,b,c] - T_target[a,b,c]) V[b, r] W[c, r]
            diff = T_current - T_target
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Normalize gradients
            grad_U /= (np.linalg.norm(grad_U) + 1e-6)
            grad_V /= (np.linalg.norm(grad_V) + 1e-6)
            grad_W /= (np.linalg.norm(grad_W) + 1e-6)
            
            # Step size
            step_size = 0.1
            
            U -= step_size * grad_U
            V -= step_size * grad_V
            W -= step_size * grad_W
            
            # Renormalize columns
            for r in range(rank):
                norm = np.linalg.norm(U[:, r])
                if norm > 1e-6:
                    U[:, r] /= norm
                    V[:, r] *= norm
                norm = np.linalg.norm(V[:, r])
                if norm > 1e-6:
                    V[:, r] /= norm
                    W[:, r] *= norm
            
            if time.time() > deadline:
                break
        
        # Check final error
        T_final = np.einsum('ar,br,cr->abc', U, V, W)
        error_final = np.linalg.norm(T_final - T_target)
        if error_final < 1e-6 and rank < best_rank:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
