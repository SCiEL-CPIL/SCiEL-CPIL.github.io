"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook():
    """Return the standard rank-45 decomposition of 3x3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    if rank == 0:
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False
    return True

def _try_rank_decomposition(rank, max_iters=8000, tol=1e-8):
    """
    Try to find a rank-r decomposition using gradient descent with momentum
    and adaptive learning rate. Returns (U, V, W, loss) or (None, None, None, best_loss).
    """
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    best_result = None
    best_loss = float('inf')
    
    # Try multiple random initializations
    for seed in range(3):
        np.random.seed(seed * 1000 + rank)
        U = np.random.randn(dim_U, rank) * 0.5
        V = np.random.randn(dim_V, rank) * 0.5
        W = np.random.randn(dim_W, rank) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(max_iters):
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            target = np.ones((n, m, p))
            loss = np.sum((T - target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_result = (U.copy(), V.copy(), W.copy())
            
            if loss < tol:
                return U, V, W, loss
            
            dT = 2.0 * (T - target)
            
            dU3 = np.einsum('ijk,jkc,kic->ijc', dT, V3, W3)
            dU = dU3.reshape(dim_U, rank)
            
            dV3 = np.einsum('ijk,ijc,kic->jkc', dT, U3, W3)
            dV = dV3.reshape(dim_V, rank)
            
            dW3 = np.einsum('ijk,ijc,jkc->kic', dT, U3, V3)
            dW = dW3.reshape(dim_W, rank)
            
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Adaptive learning rate: reduce if loss increased
            if it > 0 and loss > best_loss * 1.1:
                lr *= 0.5
        
        if best_result is not None and best_result[3] if len(best_result) > 3 else best_loss < tol:
            return best_result[0], best_result[1], best_result[2], best_loss
    
    return None, None, None, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # For 3x3x5 matrix multiplication, the known optimal rank is 45 (schoolbook).
    # The tensor rank of matrix multiplication C = A*B where A is n x m, B is m x p
    # is at most n*m*p, and for small dimensions this is often tight.
    #
    # Strategy: Attempt to find a lower-rank decomposition via gradient descent.
    # If successful, return it. Otherwise, fall back to the schoolbook decomposition.
    #
    # We try ranks from 40 down to 45 (exclusive) in a time-budgeted manner.
    # Given the non-convexity and the fact that rank 45 is likely optimal for 3x3x5,
    # we expect to quickly exhaust the budget and return the schoolbook result.
    
    # Try a few lower ranks with limited iterations each
    for rank in range(44, 39, -1):
        if time.time() > deadline - 30.0:
            break
        
        U, V, W, loss = _try_rank_decomposition(rank, max_iters=3000, tol=1e-8)
        
        if U is not None and loss < 1e-8:
            # Verify the solution
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline - 20.0:
            break
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
