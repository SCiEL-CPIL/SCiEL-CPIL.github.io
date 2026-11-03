"""Optimized program for matrix multiplication tensor decomposition 3x3x5."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

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

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False
    return True

def _try_rank_with_search(rank, max_iters=2000, tol=1e-8):
    """Try to find a rank-r decomposition using randomized search with refinement."""
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(3):
        np.random.seed(42 + attempt * 1000)
        # Initialize with small random values
        U = np.random.randn(dim_U, rank) * 0.1
        V = np.random.randn(dim_V, rank) * 0.1
        W = np.random.randn(dim_W, rank) * 0.1
        
        lr = 0.005
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
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
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
    
    if best_U is not None and best_loss < tol:
        # Round to nearest half-integer and verify
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return U_rounded, V_rounded, W_rounded, best_loss
        else:
            # Try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return U_int, V_int, W_int, best_loss
    
    return None, None, None, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Return schoolbook as fallback
    fallback = _schoolbook()
    
    # Try to find lower rank decompositions
    # Based on known results, for 3x3x3 the rank is 23, so for 3x3x5
    # we might be able to do better than 45.
    # Try a few ranks below 45
    
    for rank in [44, 43, 42, 40, 36, 32, 28]:
        if time.time() > deadline - 30:
            break
        
        U, V, W, loss = _try_rank_with_search(rank, max_iters=1500)
        
        if U is not None:
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(result["U"], result["V"], result["W"]):
                return result
    
    return fallback
