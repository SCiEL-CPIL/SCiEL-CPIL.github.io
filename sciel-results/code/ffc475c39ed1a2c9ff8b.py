import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    E = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(E, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _vectorized_search(n, m, p, target_rank, deadline, seed=42, max_restarts=10):
    """
    Vectorized gradient descent for rank-target_rank decomposition.
    """
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute E
            E = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = E - 1.0
            
            # Compute gradients
            dU = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
            dV = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
            dW = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
            
            # Update with momentum
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            
            # Optional: Check loss periodically
            if it % 50 == 0:
                if np.max(np.abs(err)) < 1e-6:
                    break
        
        # Check if we found a solution
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions with vectorized random search
    # We try ranks lower than schoolbook
    for target_rank in [60, 50, 40, 30]:
        if time.time() > deadline - 10:
            break
        candidate = _vectorized_search(n, m, p, target_rank, deadline, seed=target_rank)
        if candidate is not None:
            result = candidate
            best_rank = target_rank
            break
    
    return result
