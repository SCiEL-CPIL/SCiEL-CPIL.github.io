"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where T[i*n*m + j*m + k] = 1
    for the (i,j,k) triple. Actually the tensor is indexed such that
    T[i, j, k] = 1 for all valid i,j,k, and the decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if not np.isscalar(val):
                    val = float(np.sum(val))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _refine(U, V, W, n, m, p, max_iter=200, lr=0.01, deadline=None):
    """Iteratively refine the decomposition using ALS-style updates."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Precompute the target tensor
    T = np.ones((n, m, p), dtype=np.float64)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each row (i*m+j), U[row, :] should satisfy
        # sum_c U[row,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a linear system: U[row,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
        # Let's use gradient descent on the loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if not np.isscalar(val):
                        val = float(np.sum(val))
                    err = val - 1.0
                    loss += err * err
        if loss < 1e-12:
            break
        
        # Simple gradient descent with small step
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if not np.isscalar(val):
                        val = float(np.sum(val))
                    err = val - 1.0
                    # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                    dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                    # d/dV[j*p+k,c] = err * U[i*m+j,c] * W[k*n+i,c]
                    dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                    # d/dW[k*n+i,c] = err * U[i*m+j,c] * V[j*p+k,c]
                    dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    deadline = time.time() + 120  # safety margin
    
    # Try various ranks with random initialization
    np.random.seed(42)
    
    for rank in range(max(1, schoolbook_rank - 10), schoolbook_rank):
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Scale to get closer to target
            # Each term should contribute ~1/rank on average
            scale = (1.0 / rank) ** (1.0/3.0)
            U *= scale
            V *= scale
            W *= scale
            
            U, V, W = _refine(U, V, W, n, m, p, max_iter=100, lr=0.05, deadline=deadline)
            
            if _verify(U, V, W, n, m, p, tol=1e-4):
                if rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                break  # Found valid decomposition at this rank, try lower
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
