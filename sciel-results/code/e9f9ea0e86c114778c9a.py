"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a low-rank decomposition
    # We'll try ranks from 80 to 89, and if none work, fall back to 90
    for rank in range(80, 91):
        result = _try_decompose(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_decompose(n, m, p, rank, max_iter=100000, seed=42):
    nm, mp, pn = n * m, m * p, p * n
    np.random.seed(seed)
    
    # Initialize with random small values
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of nm*p = 90 equations
    
    # Use gradient descent on the loss: sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    lr = 0.01
    prev_loss = np.inf
    
    for it in range(max_iter):
        # Compute current values and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j]
                    vk = V[j * p + k]
                    wk = W[k * n + i]
                    val = np.sum(ui * vk * wk)
                    err = val - 1.0
                    loss += err * err
                    # d/dU[i*m+j, r] = 2*err * V[j*p+k, r] * W[k*n+i, r]
                    grad_U[i * m + j] += 2 * err * vk * wk
                    grad_V[j * p + k] += 2 * err * ui * wk
                    grad_W[k * n + i] += 2 * err * ui * vk
        
        if loss < 1e-10:
            break
        
        if abs(prev_loss - loss) < 1e-15 and it > 100:
            break
        prev_loss = loss
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Periodically try to round and check
        if it % 1000 == 0:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _check_decomposition(n, m, p, U_rounded, V_rounded, W_rounded):
                return _to_list(U_rounded, V_rounded, W_rounded)
    
    # Final check with rounding
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    if _check_decomposition(n, m, p, U_rounded, V_rounded, W_rounded):
        return _to_list(U_rounded, V_rounded, W_rounded)
    
    return None

def _check_decomposition(n, m, p, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _to_list(U, V, W):
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
