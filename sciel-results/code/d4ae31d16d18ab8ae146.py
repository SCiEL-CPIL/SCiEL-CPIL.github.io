"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.

The task is to find integer/half-integer matrices U (12 x r), V (20 x r), W (15 x r)
such that the matrix multiplication tensor is decomposed.

Strategy:
1. Start with schoolbook rank 60
2. Try to find lower rank decompositions using known constructions or search
3. Use time budget to attempt search for rank < 60

For 3x4x5, the schoolbook rank is 60. We try to find a decomposition with rank < 60.
One approach is to use Strassen-like ideas or known results for small matrix multiplication.

Given the 120s time limit, we'll:
1. Return schoolbook if we can't find better
2. Try a few constructive approaches
3. Do a limited search if time permits
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Schoolbook decomposition (rank 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition using a constructive approach
    # For 3x4x5, we can try to use block structure
    
    # Approach: Try to find a decomposition with rank < 60
    # One idea: group the 3x4 matrix into blocks and use Strassen-like ideas
    
    # Let's try a more systematic search
    # We'll use a random initialization and try to optimize
    
    # For now, let's try a few specific constructions
    
    # Construction: Try to reduce rank by sharing computations
    # For example, if we have multiple terms that share the same U or V pattern
    
    # Let's try a random search for lower rank
    # We'll initialize with random matrices and try to minimize the residual
    
    def try_random_search(target_rank, max_iter=1000):
        """Try to find a decomposition with the given rank using random search."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        
        # Initialize with random matrices
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Try to optimize using gradient descent
        # The objective is to minimize the residual:
        # residual[i,j,k] = 1 - sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        lr = 0.01
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute residual for this (i,j,k)
                        residual = 1.0
                        for r in range(target_rank):
                            residual -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        
                        # Gradient contribution
                        for r in range(target_rank):
                            # d/dU[i*m+j,r] of -sum U*V*W
                            grad_U[i*m+j, r] += residual * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += residual * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += residual * U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if iteration % 100 == 0:
                max_residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                            max_residual = max(max_residual, abs(val - 1.0))
                if max_residual < 1e-6:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try to find a decomposition with rank < 60
    # Start with rank 55 and go down
    for target_rank in range(55, 30, -5):
        if time.time() > deadline:
            break
        
        result = try_random_search(target_rank, max_iter=500)
        if result is not None:
            U = np.array(result["U"])
            V = np.array(result["V"])
            W = np.array(result["W"])
            if check_decomposition(U, V, W):
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
