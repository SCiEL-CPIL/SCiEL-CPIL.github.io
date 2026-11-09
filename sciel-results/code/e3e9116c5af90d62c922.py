"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a randomized approach to find a low-rank decomposition.
    The key insight is that we can try to find a decomposition with rank less than n*m*p = 36.
    
    For matrix multiplication, the bilinear complexity (rank) of multiplying an m x n matrix by an n x p matrix
    is known to be at most m*n*p, but can often be lower.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 36)
    2. Try to find a lower rank decomposition using random initialization and optimization
    3. Use the structure of matrix multiplication to guide the search
    
    The tensor for matrix multiplication A (2x3) * B (3x6) = C (2x6) is:
    T[i, j, k, l] = 1 if and only if the product C[i,l] = sum_j A[i,j] * B[j,l]
    
    In the factorized form:
    U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] should reconstruct the multiplication
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Initialize with schoolbook
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Use random restarts with different initial ranks
    
    # The minimum possible rank for this problem is at least max(n*p, m*p, n*m) = max(12, 18, 6) = 18
    # But we need to find an exact decomposition
    
    # Try ranks from 18 up to 35
    for target_rank in range(18, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize columns to avoid scaling issues
            for r in range(target_rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                    U[:, r] /= norm_U
                    V[:, r] /= norm_V
                    W[:, r] /= norm_W
            
            # Compute the reconstructed tensor and compare to the true tensor
            # The true tensor T[i,j,k,l] = 1 if i,j,k,l are valid indices for matrix multiplication
            
            # For each rank component r, we have:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal 1 when (i,j,k) corresponds to a valid multiplication step
            
            # Let's check the decomposition by computing the implied multiplication
            # C[i,l] = sum_j A[i,j] * B[j,l]
            # In the factorized form, this becomes:
            # C[i,l] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+l, r] * B[j,l]) * W[l*n+i, r]
            
            # Actually, let me think about this differently.
            # The standard factorization for matrix multiplication is:
            # C[i,k] = sum_r U[i, r] * V[r, k] where U and V are derived from A and B
            # But here we have a different structure.
            
            # Let me use the standard approach:
            # For matrix multiplication C = A * B where A is n x m and B is m x p:
            # C[i, k] = sum_j A[i, j] * B[j, k]
            
            # The tensor decomposition is:
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[i, k, r]
            # where T[i, j, k] = 1 (the multiplication tensor)
            
            # In our flattened form:
            # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
            # And 0 otherwise (but all (i,j,k) with valid indices are part of the tensor)
            
            # Let's verify by computing the sum for all (i,j,k)
            error = 0.0
            max_error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1.0) ** 2
                        max_error = max(max_error, abs(val - 1.0))
            
            if max_error < 1e-6:
                # Found a valid decomposition!
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                # Convert to lists and return
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
            
            # Try gradient descent to minimize the error
            if time.time() > deadline:
                break
            
            # Simple gradient descent
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            grad = 2 * (val - 1.0)
                            for r in range(target_rank):
                                dU[i * m + j, r] += grad * V[j * p + k, r] * W[k * n + i, r]
                                dV[j * p + k, r] += grad * U[i * m + j, r] * W[k * n + i, r]
                                dW[k * n + i, r] += grad * U[i * m + j, r] * V[j * p + k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Recompute error
                max_error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            max_error = max(max_error, abs(val - 1.0))
                
                if max_error < 1e-6:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
    
    # If no lower rank found, return the schoolbook decomposition
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
