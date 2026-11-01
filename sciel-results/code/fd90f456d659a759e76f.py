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
        
        # Check all conditions
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                actual = np.dot(U[i * m + j], V[j_prime * p + k] * W[i_prime * n + k_prime])
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    def compute_residual(U, V, W):
        """Compute the squared residual of the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        residual = 0.0
        
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                actual = np.dot(U[i * m + j], V[j_prime * p + k] * W[i_prime * n + k_prime])
                                residual += (actual - expected) ** 2
        
        return residual
    
    def gradient_descent(U, V, W, max_iters=500, lr=0.001, tol=1e-8):
        """Perform gradient descent to minimize the decomposition residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        rank = U.shape[1]
        
        best_residual = compute_residual(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients numerically for stability
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            eps = 1e-7
            
            # Gradient for U
            for i in range(n):
                for i_prime in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k in range(p):
                                for k_prime in range(p):
                                    expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                    actual = np.dot(U[i * m + j], V[j_prime * p + k] * W[i_prime * n + k_prime])
                                    grad_factor = 2.0 * (actual - expected)
                                    dU[i * m + j] += grad_factor * (V[j_prime * p + k] * W[i_prime * n + k_prime])
            
            # Gradient for V
            for i in range(n):
                for i_prime in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k in range(p):
                                for k_prime in range(p):
                                    expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                    actual = np.dot(U[i * m + j], V[j_prime * p + k] * W[i_prime * n + k_prime])
                                    grad_factor = 2.0 * (actual - expected)
                                    dV[j_prime * p + k] += grad_factor * (U[i * m + j] * W[i_prime * n + k_prime])
            
            # Gradient for W
            for i in range(n):
                for i_prime in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k in range(p):
                                for k_prime in range(p):
                                    expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                    actual = np.dot(U[i * m + j], V[j_prime * p + k] * W[i_prime * n + k_prime])
                                    grad_factor = 2.0 * (actual - expected)
                                    dW[i_prime * n + k_prime] += grad_factor * (U[i * m + j] * V[j_prime * p + k])
            
            # Update with adaptive learning rate
            norm = np.linalg.norm(dU) + np.linalg.norm(dV) + np.linalg.norm(dW)
            if norm > 1e-12:
                scale = lr / norm
                U -= scale * dU
                V -= scale * dV
                W -= scale * dW
            
            residual = compute_residual(U, V, W)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_residual < tol:
                break
        
        return best_U, best_V, best_W, best_residual
    
    def try_rank(target_rank, num_restarts=5, max_iters=200):
        """Try to find a valid decomposition of the given rank."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_residual = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization based on schoolbook
            np.random.seed(restart * 1000 + target_rank)
            
            # Start from a perturbed schoolbook-like structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random but structured values
            for r in range(target_rank):
                # Assign random "active" indices
                i_idx = np.random.randint(0, n)
                j_idx = np.random.randint(0, m)
                k_idx = np.random.randint(0, p)
                
                U[i_idx * m + j_idx, r] = 1.0
                V[j_idx * p + k_idx, r] = 1.0
                W[k_idx * n + i_idx, r] = 1.0
                
                # Add small random perturbations to other entries
                U[:, r] += np.random.randn(n * m) * 0.1
                V[:, r] += np.random.randn(m * p) * 0.1
                W[:, r] += np.random.randn(p * n) * 0.1
            
            # Normalize
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
            if check_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), tol=1e-5):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
        # If the best residual is small, check again with tighter tolerance
        if best_result is not None and best_residual < 1e-4:
            U_opt, V_opt, W_opt = best_result
            if check_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), tol=1e-6):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
        return None
    
    # Start with schoolbook as fallback
    best_decomp = schoolbook_decomposition()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 2x3 x 3x7, try from 35 down to 25
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_decomp = result
            best_rank = target_rank
            break
    
    return best_decomp
# EVOLVE-BLOCK-END

