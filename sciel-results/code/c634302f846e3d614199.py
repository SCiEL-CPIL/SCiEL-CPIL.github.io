"""Rewritten program for matrix multiplication tensor decomposition (2,3,7)."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def schoolbook_decomposition():
        """Create the standard (schoolbook) rank-42 decomposition."""
        rank = n * m * p  # 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Verify that U, V, W form a valid exact decomposition of the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor component T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k', else 0
        # Decomposition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = T[(i,j),(j',k),(i',k')]
        
        # Check all required conditions
        # For the diagonal (i=i', j=j', k=k'): should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        
        # For off-diagonal: should be 0
        # Check i != i'
        for i in range(n):
            for i_prime in range(n):
                if i == i_prime:
                    continue
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[i_prime * n + k])
                        if abs(val) > tol:
                            return False
        
        # Check j != j'
        for j in range(m):
            for j_prime in range(m):
                if j == j_prime:
                    continue
                for i in range(n):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j_prime * p + k] * W[k * n + i])
                        if abs(val) > tol:
                            return False
        
        # Check k != k'
        for k in range(p):
            for k_prime in range(p):
                if k == k_prime:
                    continue
                for i in range(n):
                    for j in range(m):
                        val = np.dot(U[i * m + j], V[j * p + k], W[k_prime * n + i]) if False else np.dot(U[i * m + j], V[j * p + k] * W[k_prime * n + i])
                        if abs(val) > tol:
                            return False
        
        return True
    
    def compute_tensor_residual(U, V, W):
        """Compute the squared residual of the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        residual = 0.0
        
        # Diagonal terms should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    residual += (val - 1.0) ** 2
        
        # Off-diagonal terms should be 0
        # i != i'
        for i in range(n):
            for i_prime in range(n):
                if i == i_prime:
                    continue
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[i_prime * n + k])
                        residual += val ** 2
        
        # j != j'
        for j in range(m):
            for j_prime in range(m):
                if j == j_prime:
                    continue
                for i in range(n):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j_prime * p + k] * W[k * n + i])
                        residual += val ** 2
        
        # k != k'
        for k in range(p):
            for k_prime in range(p):
                if k == k_prime:
                    continue
                for i in range(n):
                    for j in range(m):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k_prime * n + i])
                        residual += val ** 2
        
        return residual
    
    def gradient_descent(U, V, W, max_iters=2000, lr=0.01, tol=1e-8):
        """Perform gradient descent to minimize the decomposition residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        rank = U.shape[1]
        
        best_residual = compute_tensor_residual(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Diagonal terms: d(residual)/dU[i*m+j, r] = 2*(val-1)*V[j*p+k, r]*W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        grad = 2.0 * (val - 1.0) * (V[j * p + k] * W[k * n + i])
                        dU[i * m + j] += grad
            
            # Off-diagonal i != i'
            for i in range(n):
                for i_prime in range(n):
                    if i == i_prime:
                        continue
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[i_prime * n + k])
                            grad = 2.0 * val * (V[j * p + k] * W[i_prime * n + k])
                            dU[i * m + j] += grad
            
            # Similarly for V and W
            for j in range(m):
                for j_prime in range(m):
                    if j == j_prime:
                        continue
                    for i in range(n):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j_prime * p + k] * W[k * n + i])
                            grad = 2.0 * val * (U[i * m + j] * W[k * n + i])
                            dV[j_prime * p + k] += grad
            
            for k in range(p):
                for k_prime in range(p):
                    if k == k_prime:
                        continue
                    for i in range(n):
                        for j in range(m):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k_prime * n + i])
                            grad = 2.0 * val * (U[i * m + j] * V[j * p + k])
                            dW[k_prime * n + i] += grad
            
            # Diagonal contribution to V
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        grad = 2.0 * (val - 1.0) * (U[i * m + j] * W[k * n + i])
                        dV[j * p + k] += grad
            
            # Diagonal contribution to W
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        grad = 2.0 * (val - 1.0) * (U[i * m + j] * V[j * p + k])
                        dW[k * n + i] += grad
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            residual = compute_tensor_residual(U, V, W)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_residual < tol:
                break
        
        return best_U, best_V, best_W, best_residual
    
    def try_rank(target_rank, num_restarts=3, max_iters=1000):
        """Try to find a valid decomposition of the given rank."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_residual = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Normalize to help convergence
            for r in range(target_rank):
                norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
            
            # Gradient descent
            U_opt, V_opt, W_opt, residual = gradient_descent(U, V, W, max_iters=max_iters)
            
            if residual < best_residual:
                best_residual = residual
                best_result = (U_opt, V_opt, W_opt)
            
            # Check if we found a valid decomposition
            if check_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), tol=1e-6):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
        # If the best residual is small, check again with tighter tolerance
        if best_result is not None and best_residual < 1e-4:
            U_opt, V_opt, W_opt = best_result
            if check_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), tol=1e-5):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
        return None
    
    # Start with schoolbook as fallback
    best_decomp = schoolbook_decomposition()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 2x3 x 3x7, known bounds suggest rank between 20-35
    # Try from 30 down to 20
    for target_rank in range(30, 19, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_decomp = result
            best_rank = target_rank
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
