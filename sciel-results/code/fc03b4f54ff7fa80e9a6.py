"""Improved program for matrix multiplication 2x5*5x6 tensor rank search."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin (150s limit, leave 35s buffer)
    
    # Schoolbook decomposition as baseline (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is valid using the tensor convention."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Vectorized check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-8):
        """Faster verification using matrix operations."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is a 3-index tensor of shape (n, m, p)
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            # U[:,c] has shape (n*m,), reshape to (n,m)
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            # T[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
            for i in range(n):
                for j in range(m):
                    # T[i,j,:] += U_c[i,j] * (V_c[j,:] * W_c[:,i])
                    T[i, j, :] += U_c[i, j] * (V_c[j, :] * W_c[:, i])
        return np.allclose(T, 1.0, atol=tol)
    
    def compute_residual(U, V, W):
        """Compute the residual tensor T - 1."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += U_c[i, j] * (V_c[j, :] * W_c[:, i])
        return T - 1.0
    
    # Strategy: Use a more structured approach
    # Key insight: For 2x5*5x6, we can try to exploit the small n=2 dimension
    
    # Approach: Parameterize the problem and use optimization
    # The tensor equation is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try a different strategy: use the known structure
    # C = A*B where A is 2x5, B is 5x6
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as a bilinear form. The rank is the minimum number of
    # scalar multiplications needed.
    
    # Let's try to find a decomposition with rank < 60 using a structured search.
    
    # Key idea: Use the fact that we can combine terms.
    # For example, consider:
    # (A[0,j] + A[1,j]) * B[j,k] gives us A[0,j]*B[j,k] + A[1,j]*B[j,k]
    # This saves one multiplication per (j,k) pair.
    
    # More systematically, let's try to find a set of "multipliers" that
    # allow us to express all 60 products using fewer than 60 scalar multiplications.
    
    # Let's use a gradient-based approach on a smooth surrogate.
    # Define F(U,V,W) = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    # We want to minimize F to 0.
    
    rng = np.random.default_rng(12345)
    
    # Try various ranks and use a structured initialization
    for target_rank in range(59, 44, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random restarts
        max_restarts = 15
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a structured pattern
            # Start from schoolbook and randomly merge columns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # First, fill with a base pattern
            # Use a cyclic assignment
            for c in range(target_rank):
                # Assign each column to a subset of (i,j,k) triples
                # Use a hash-like function to distribute
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i * m * p + j * p + k) % target_rank
                            if idx == c:
                                U[i * m + j, c] = 1.0
                                V[j * p + k, c] = 1.0
                                W[k * n + i, c] = 1.0
            
            # Add some random perturbation
            scale = 0.1
            U += rng.standard_normal((n * m, target_rank)) * scale
            V += rng.standard_normal((m * p, target_rank)) * scale
            W += rng.standard_normal((p * n, target_rank)) * scale
            
            # Try to verify
            if verify_fast(U, V, W):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
            
            # If not verified, try a local optimization step
            # Use a simple gradient descent on the residual
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                
                residual = compute_residual(U, V, W)
                if np.max(np.abs(residual)) < 1e-8:
                    if verify_fast(U, V, W):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
                    break
                
                # Compute gradients (approximate)
                lr = 0.001
                # Gradient of F w.r.t. U[i*m+j,c]:
                # dF/dU[i*m+j,c] = 2 * sum_{i',j',k'} residual[i',j',k'] * delta(i',i)*delta(j',j) * V[j*p+k,c]*W[k*n+i,c]
                # This is complex, so let's use finite differences for a few steps
                
                # Actually, let's try a different approach: use the structure more directly
                break
            
            if found:
                break
        
        if found:
            break
    
    # If we haven't found a better decomposition, try one more structured approach
    if best_rank == rank_school:
        # Try to use the "sum trick" explicitly
        # For each j,k, instead of computing A[0,j]*B[j,k] and A[1,j]*B[j,k] separately,
        # compute (A[0,j]+A[1,j])*B[j,k] and (A[0,j]-A[1,j])*B[j,k]
        # Then A[0,j]*B[j,k] = 0.5*((sum)+ (diff)) and A[1,j]*B[j,k] = 0.5*((sum)-(diff))
        # This uses 2 multiplications instead of 2, so no savings directly.
        # But if we can share across different (i,j,k) combinations, we might save.
        
        # Let's try a completely different structured approach:
        # Use the fact that the tensor is a sum of rank-1 tensors.
        # Each term in the sum corresponds to a scalar multiplication.
        
        # Try to find a sparse solution using a greedy approach
        pass
    
    return best_result
# EVOLVE-BLOCK-END
