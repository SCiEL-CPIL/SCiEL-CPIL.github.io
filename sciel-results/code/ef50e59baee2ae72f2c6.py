import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    # Try to find a better decomposition
    # We know rank <= 64
    # Let's try to find a rank 60 decomposition using random search
    # This is a heuristic and might fail
    
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Target rank to try
    target_ranks = [60, 62, 64]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition with this rank
        # We can use a simple iterative method
        # Initialize randomly
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this decomposes the tensor
            # The tensor T[i, j, k] = delta(i, j, k)
            # We need sum_c U[i, c] V[j, c] W[k, c] = delta(i, j, k)
            # Actually, the tensor is defined by U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # So we need to check if the product matches the multiplication tensor
            
            # Construct the tensor from U, V, W
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is expensive to compute fully.
            # We can check a few random points or use a linear algebra trick.
            
            # Let's check the error on a subset of points
            # Or check the full tensor if it's small
            # n=3, m=3, p=8. Total points 72.
            # We can compute the full tensor and check for zeros.
            
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum
                        val = 0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        
                        # Expected value is 1 if i==j? No.
                        # The tensor for matrix multiplication is:
                        # C[i, k] = sum_j A[i, j] B[j, k]
                        # The tensor T[i, j, k] is 1 if the indices match the multiplication structure.
                        # Specifically, T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k].
                        # So T[i, j, k] is 1 for all i, j, k?
                        # No, the tensor is the structure of the bilinear form.
                        # The bilinear form is sum_{i, j, k} T[i, j, k] A[i, j] B[j, k] = C[i, k].
                        # So T[i, j, k] is 1 if the indices (i, j, k) correspond to a term in the sum for C[i, k].
                        # For a fixed i, k, the sum is over j.
                        # So T[i, j, k] = 1 for all i, j, k?
                        # No, T is a 3-tensor.
                        # The condition is:
                        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if (i, j, k) is a valid term?
                        # Actually, the decomposition satisfies:
                        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i, i} delta_{k, k}?
                        # No.
                        # The tensor is defined such that:
                        # C[i, k] = sum_j A[i, j] B[j, k]
                        # = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i, j] B[j, k]?
                        # No, the decomposition is:
                        # A[i, j] B[j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] ?
                        # No, that's not right.
                        
                        # The standard decomposition is:
                        # U, V, W are matrices.
                        # The condition is:
                        # sum_c U[i, c] V[j, c] W[k, c] = T[i, j, k]
                        # where T[i, j, k] is the multiplication tensor.
                        # For matrix multiplication, T[i, j, k] is 1 if i, j, k form a valid path?
                        # Actually, T[i, j, k] = 1 for all i, j, k?
                        # No, T is sparse.
                        # T[i, j, k] = 1 if the indices correspond to a term in the multiplication.
                        # For M_{n, m, p}, T[i, j, k] = 1 if i < n, j < m, k < p.
                        # Wait, the tensor is defined on indices i (0..n-1), j (0..m-1), k (0..p-1).
                        # The value T[i, j, k] is 1 if we are multiplying A[i, j] and B[j, k] to get C[i, k].
                        # So T[i, j, k] = 1 for all i, j, k?
                        # No, because C[i, k] is the sum over j.
                        # So the tensor is sum_j A[i, j] B[j, k].
                        # The bilinear form is sum_{i, j, k} T[i, j, k] A[i, j] B[j, k].
                        # So T[i, j, k] = 1 for all i, j, k?
                        # Yes, because every term A[i, j] B[j, k] contributes to C[i, k].
                        # So T[i, j, k] = 1 for all i, j, k.
                        
                        # Wait, if T[i, j, k] = 1 for all i, j, k, then the rank is n*m*p.
                        # And the schoolbook decomposition has rank n*m*p.
                        # So the condition is:
                        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
                        
                        if val != 1:
                            error += abs(val - 1)
                            
            if error < 1e-9:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
                
    # Return the best found
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
