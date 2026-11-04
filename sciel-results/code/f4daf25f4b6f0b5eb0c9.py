"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s
    
    # Try to find a lower-rank decomposition
    # The tensor for matrix multiplication has entries T[i,j,k] = A[i,j]*B[j,k]
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But better bounds exist; let's try from 30 upward
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, store the schoolbook as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions using optimization
    # We'll use a simple approach: random init + gradient descent on the loss
    
    def tensor_residual(U, V, W, rank):
        """Compute the residual between the tensor decomposition and the true tensor."""
        # True tensor: T[i, j, k] = 1 if we consider A[i,j]*B[j,k]
        # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal 1 for all (i,j,k)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual += (val - 1.0) ** 2
        return residual
    
    def compute_gradients(U, V, W, rank):
        """Compute gradients of the squared residual w.r.t. U, V, W."""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Current value
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = val - 1.0
                    
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * err * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * err * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * err * U[i * m + j, r] * V[j * p + k, r]
        
        return dU, dV, dW
    
    # Try various ranks
    for target_rank in range(30, min(schoolbook_rank, 50)):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + restart * 1000 + target_rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Set some entries to 1 to help convergence
            # Use the schoolbook pattern as a hint
            for r in range(min(target_rank, n * m * p)):
                # Map to schoolbook indices
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                if i < n and j < m and k < p:
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
            
            # Gradient descent
            lr = 0.01
            prev_loss = tensor_residual(U, V, W, target_rank)
            max_iters = 500
            
            for it in range(max_iters):
                if time.time() > deadline - 30:
                    break
                
                dU, dV, dW = compute_gradients(U, V, W, target_rank)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 50 == 0:
                    curr_loss = tensor_residual(U, V, W, target_rank)
                    if curr_loss < 1e-10:
                        found = True
                        break
                    if curr_loss > prev_loss:
                        lr *= 0.5
                    prev_loss = curr_loss
            
            if found:
                break
        
        if found:
            # Verify the solution
            final_loss = tensor_residual(U, V, W, target_rank)
            if final_loss < 1e-8:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
    
    if best_result is not None:
        return best_result
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
# EVOLVE-BLOCK-END
