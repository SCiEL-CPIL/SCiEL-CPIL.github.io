"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses gradient-based optimization to find low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    target_time = time.time() + 110  # Leave 10s buffer
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition is:
    # sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    def is_exact(U, V, W, n, m, p, tol=1e-7):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def to_half_integers(U, V, W, tol=1e-6):
        """Try to snap values to nearest half-integers if close."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if (np.max(np.abs(U - U_snapped)) < tol and
            np.max(np.abs(V - V_snapped)) < tol and
            np.max(np.abs(W - W_snapped)) < tol):
            return U_snapped, V_snapped, W_snapped
        return None
    
    def optimize_decomposition(r, init_seed=42, max_iter=5000):
        """
        Use gradient descent to find a rank-r decomposition.
        We minimize the sum of squared errors over all (i,j,k) entries.
        """
        np.random.seed(init_seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Learning rate schedule
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > target_time:
                break
            
            # Compute loss and gradients
            # L = sum_{i,j,k} (sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1)^2
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute the current value
                        val = 0.0
                        for t in range(r):
                            val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                        err = val - 1.0
                        loss += err * err
                        
                        # Gradients
                        for t in range(r):
                            # dL/dU[i*m+j,t] = 2*err * V[j*p+k,t] * W[k*n+i,t]
                            grad_U[i * m + j, t] += 2 * err * V[j * p + k, t] * W[k * n + i, t]
                            # dL/dV[j*p+k,t] = 2*err * U[i*m+j,t] * W[k*n+i,t]
                            grad_V[j * p + k, t] += 2 * err * U[i * m + j, t] * W[k * n + i, t]
                            # dL/dW[k*n+i,t] = 2*err * U[i*m+j,t] * V[j*p+k,t]
                            grad_W[k * n + i, t] += 2 * err * U[i * m + j, t] * V[j * p + k, t]
            
            # Update parameters
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if iteration % 500 == 499:
                lr *= 0.95
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Early stopping if we found an exact solution
            if loss < 1e-10:
                break
        
        return best_U, best_V, best_W, best_loss
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But we can try lower ranks too
    
    schoolbook_rank = n * m * p
    
    # Try to find the minimum rank that works
    for r in range(1, schoolbook_rank + 1):
        if time.time() > target_time:
            break
        
        # Try multiple initializations for this rank
        found = False
        for seed in range(5):
            if time.time() > target_time:
                break
            
            U, V, W, loss = optimize_decomposition(r, init_seed=seed + 100)
            
            if loss < 1e-8:
                # Check if it's exact with half-integers
                snapped = to_half_integers(U, V, W)
                if snapped is not None:
                    U, V, W = snapped
                    if is_exact(U, V, W, n, m, p):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                else:
                    # Check if it's exact as-is
                    if is_exact(U, V, W, n, m, p):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        # If we found a solution for this rank, return it
        if found:
            break
    
    # Fallback: return schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
