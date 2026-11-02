import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def schoolbook():
        rank = n * m * p
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
        return U, V, W
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T (within tolerance)"""
        # Compute sum_r U[:,r] V[:,r]^T W[:,r]^T
        # This is a tensor of shape (n*m, m*p, p*n)
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def to_half_integer(U, V, W):
        """Round to nearest half-integer and verify"""
        U_hi = np.round(U * 2) / 2
        V_hi = np.round(V * 2) / 2
        W_hi = np.round(W * 2) / 2
        if verify(U_hi, V_hi, W_hi, tol=1e-9):
            return U_hi, V_hi, W_hi
        return None
    
    def try_rank(rank, max_iter=5000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent"""
        if time.time() > deadline:
            return None
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute current approximation
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_approx - T) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dL/dU[a,r] = 2 * sum_{b,c} (T_approx[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} (T_approx[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} (T_approx[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            diff = T_approx - T  # shape (n*m, m*p, p*n)
            
            dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Try to round to half-integers
        result = to_half_integer(best_U, best_V, best_W)
        if result is not None:
            return result
        
        # Also try with the original (unrounded) if loss is very small
        if best_loss < 1e-8:
            # Try different rounding strategies
            for offset in [0, 0.25, 0.5, 0.75]:
                U_off = np.round((best_U + offset) * 2) / 2
                V_off = np.round((best_V + offset) * 2) / 2
                W_off = np.round((best_W + offset) * 2) / 2
                if verify(U_off, V_off, W_off, tol=1e-9):
                    return U_off, V_off, W_off
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But for matrix multiplication, Strassen-like methods can do better than n*m*p
    # For 3x4 x 4x5, let's try ranks from 25 up to 59
    
    best_result = None
    for rank in range(25, min(n * m * p, 60)):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            break  # Found a valid decomposition at this rank
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list format as expected
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
