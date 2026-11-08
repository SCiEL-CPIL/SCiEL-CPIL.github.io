"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 4*4*9 = 144
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with rank less than schoolbook
    # The minimum rank for 4x4 * 4x9 is known to be at most 144 (schoolbook)
    # but could be lower. Let's try to find a lower rank decomposition.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p,
    # the tensor has entries T[i,j,k,l] = delta_{j,l} * delta_{i,k} ... 
    # Actually, the multiplication tensor T has components T[i,k,j] for A[i,j]*B[j,k]
    # The standard decomposition is rank n*m*p.
    
    # Let's use a randomized search approach to find a lower rank decomposition
    # or verify if schoolbook is optimal for this case.
    
    # First, let's check known bounds:
    # For 4x4 * 4x9, the minimum rank is not known to be less than 144 in general,
    # but let's try a few approaches.
    
    # Approach 1: Try random rank-r decompositions and check if they work
    # The multiplication tensor M has entries M[i,k,j] = 1 if we're multiplying A[i,j]*B[j,k]
    # A rank-r decomposition gives: M[i,k,j] = sum_{c=1}^r U[i,c] * V[k,c] * W[j,c]
    # Wait, let me re-read the convention.
    
    # From the task: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) where k is from B's columns and i is from A's rows
    # The decomposition should satisfy: for all i, j, k:
    #   delta_{i,i}delta_{j,j}delta_{k,k} ... no
    #   The (i,k) entry of A*B is sum_j A[i,j] B[j,k]
    #   So the tensor T[i,j,k] = 1 (it's the identity in the middle index)
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify schoolbook works:
    # For schoolbook, each term c corresponds to (i,j,k) triple
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if there's a c matching (i,j,k), else 0
    # That gives T[i,j,k] = 1 for all i,j,k, which is correct.
    
    # Now let's try to find a lower rank decomposition using optimization.
    # We'll use a simple iterative approach.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, keep safety margin
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But actual lower bound is higher. Let's try a few ranks.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Generate schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank decompositions
    # Use a randomized gradient-free approach
    for target_rank in range(36, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Define the error function
            def error(U, V, W):
                # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # and compare with 1
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0) ** 2
                return err
            
            # Simple gradient descent
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradients numerically
                eps = 1e-5
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # This is too slow for numerical gradients, let's use analytic
                # dE/dU[i*m+j,c] = 2 * sum_{i,j,k} (val[i,j,k] - 1) * V[j*p+k,c] * W[k*n+i,c]
                # where val[i,j,k] = sum_c' U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                
                # Precompute val
                val = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                
                diff = val - 1.0
                
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            grad_U[i*m+j, c] = 2.0 * np.sum(diff[i, j, :] * V[j*p+np.arange(p), c] * W[np.arange(p)*n+i, c])
                    for j in range(m):
                        for k in range(p):
                            grad_V[j*p+k, c] = 2.0 * np.sum(diff[:, j, k] * U[np.arange(n)*m+j, c] * W[k*n+np.arange(n), c])
                    for k in range(p):
                        for i in range(n):
                            grad_W[k*n+i, c] = 2.0 * np.sum(diff[i, :, k] * U[i*m+np.arange(m), c] * V[np.arange(m)*p+k, c])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if step % 10 == 0:
                    err = error(U, V, W)
                    if err < 1e-6:
                        # Found a good solution
                        # Convert to list format
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
