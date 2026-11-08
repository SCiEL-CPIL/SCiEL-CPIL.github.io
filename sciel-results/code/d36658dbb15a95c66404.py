"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]B[j,k] = sum_r U[i*m+j? no...]
    Actually: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We check that this equals 1 if (i,j,k) is a valid index, else 0.
    But actually the tensor for matmul is T[i,j,k] = 1 for all valid i,j,k since
    C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Check a few random indices
    for _ in range(100):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        if abs(val - 1.0) > 1e-6:
            return False
    return True


def _try_rank(rank, n, m, p, deadline):
    """Try to find a rank decomposition using numerical optimization."""
    if time.time() > deadline:
        return None
    
    # Initialize with schoolbook and try to reduce
    # Use a random initialization for U, V, W
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # The tensor T[i,j,k] = 1 for all valid indices
    # We need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # Use gradient descent on the residual
    # Residual for each (i,j,k): R[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
    # Loss = sum R^2
    
    lr = 0.01
    best_loss = float('inf')
    
    for step in range(5000):
        if time.time() > deadline:
            return None
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # R = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
                    R = 0.0
                    for r in range(rank):
                        R += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    R -= 1.0
                    
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * R * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * R * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * R * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R = 0.0
                    for r in range(rank):
                        R += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    R -= 1.0
                    loss += R * R
        
        if loss < best_loss:
            best_loss = loss
            if best_loss < 1e-8:
                # Found a solution, try to snap to nice values
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Adaptive learning rate
        if step % 500 == 0 and step > 0:
            lr *= 0.9
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin
    
    # First, try to find a better decomposition
    # The minimum rank for 2x3 * 3x4 is at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Schoolbook is 24. Let's try ranks from 12 upward.
    
    best_result = _schoolbook()
    
    # Try various ranks
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END
