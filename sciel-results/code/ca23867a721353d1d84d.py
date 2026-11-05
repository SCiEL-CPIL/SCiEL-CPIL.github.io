import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The target tensor T[i,j,k] = 1 for all i,j,k
    # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_with_optimization(n, m, p, rank, max_iterations=500, tol=1e-8, time_limit=10):
    """Try to find a rank-r decomposition using random init + gradient descent."""
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Gradient descent optimization
        lr = 0.01
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current values
            errors = []
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        errors.append(err**2)
                        
                        # Gradients
                        dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
            
            loss = np.mean(errors)
            
            # Check convergence
            if loss < tol:
                return U, V, W, loss
            
            # Update with adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 1e-10:
                step = lr / (1 + it * 0.01)
                U -= step * dU
                V -= step * dV
                W -= step * dW
        
        # Final check
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < tol:
            return U, V, W, max_err
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    deadline = time.time() + 110
    
    # Try ranks from 40 up to schoolbook
    best_result = None
    best_rank = schoolbook_rank
    
    for rank in range(40, min(schoolbook_rank, 70)):
        if time.time() > deadline:
            break
        
        # Try to find solution at this rank
        result = _try_rank_with_optimization(n, m, p, rank, max_iterations=200, tol=1e-8, time_limit=8)
        
        if result is not None:
            U, V, W, loss = result
            # Verify
            if _verify(U, V, W, n, m, p, rank, tol=1e-7):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break  # Found lowest rank, stop
    
    if best_result is not None:
        return best_result
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
