import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Helper to verify if a decomposition is exact
    def is_exact(U, V, W):
        # Check if U @ diag(1) @ V^T matches the tensor for each component of W
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 if i,j,k are valid indices for the multiplication?
        # No, the tensor for matrix multiplication C = AB is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T_{i,j,k} such that C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k]
        # So T_{i,j,k} = 1 for all valid i,j,k.
        
        # We need to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
        # And 0 otherwise? No, the indices are fixed.
        
        # Let's compute the full tensor and check against the target tensor.
        # Target tensor: T[i,j,k] = 1
        
        # Compute reconstructed tensor
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This can be done efficiently?
        # U shape: (n*m, r)
        # V shape: (m*p, r)
        # W shape: (p*n, r)
        
        # Let's just iterate for small dimensions since we are checking exactness.
        # n=2, m=3, p=9. Total elements = 54.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank
    # We will try ranks from 50 down to 40
    for r in range(50, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize random decomposition
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Local search
            # We need a gradient or some heuristic.
            # Since we need exact integer/half-integer solutions, random search in continuous space
            # followed by rounding might work if the landscape is smooth enough, but tensor rank minimization
            # is non-convex and hard.
            
            # Let's try a simple local search:
            # 1. Compute gradient of sum (T_recon - 1)^2
            # 2. Update U, V, W
            
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Compute gradient
                # L = sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
                # dL/dU[i,j,r] = 2 * (sum_{i,j,k} (T_recon[i,j,k] - 1) * V[j,k,r] * W[k,i,r])
                
                # This is computationally expensive to do in pure Python loops.
                # Let's use NumPy to vectorize the error computation.
                
                # We can reshape U, V, W to 3D tensors?
                # U is (n*m, r). Reshape to (n, m, r).
                # V is (m*p, r). Reshape to (m, p, r).
                # W is (p*n, r). Reshape to (p, n, r).
                
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                # T_recon[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                
                # This contraction is not standard.
                # Let's compute T_recon as a 3D array.
                T_recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i,j,k] = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                
                error = T_recon - 1.0
                if np.max(np.abs(error)) < 1e-9:
                    # Found exact solution!
                    # Round to nearest 0.5
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    if is_exact(U, V, W):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = r
                        # We can break out of loops to save time or continue to find lower rank
                        # But since we are iterating r downwards, we should continue to try lower r
                        # But we found one for this r.
                        break
                
                # Compute gradients
                # dL/dU[i,j,r] = 2 * sum_{k} error[i,j,k] * V3[j,k,r] * W3[k,i,r]
                # dL/dV[j,k,r] = 2 * sum_{i} error[i,j,k] * U3[i,j,r] * W3[k,i,r]
                # dL/dW[k,i,r] = 2 * sum_{j} error[i,j,k] * U3[i,j,r] * V3[j,k,r]
                
                grad_U = np.zeros((n, m, r))
                grad_V = np.zeros((m, p, r))
                grad_W = np.zeros((p, n, r))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i,j,k]
                            if e != 0:
                                # grad_U[i,j,:] += 2 * e * V3[j,k,:] * W3[k,i,:]
                                grad_U[i,j,:] += 2 * e * V3[j,k,:] * W3[k,i,:]
                                # grad_V[j,k,:] += 2 * e * U3[i,j,:] * W3[k,i,:]
                                grad_V[j,k,:] += 2 * e * U3[i,j,:] * W3[k,i,:]
                                # grad_W[k,i,:] += 2 * e * U3[i,j,:] * V3[j,k,:]
                                grad_W[k,i,:] += 2 * e * U3[i,j,:] * V3[j,k,:]
                
                # Update
                lr = 0.01
                U -= lr * grad_U.reshape(n*m, r)
                V -= lr * grad_V.reshape(m*p, r)
                W -= lr * grad_W.reshape(p*n, r)
            
            # If we found a valid decomposition, update best
            # (Checked inside loop)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
