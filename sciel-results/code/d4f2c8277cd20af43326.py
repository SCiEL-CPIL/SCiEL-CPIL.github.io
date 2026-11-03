import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify if U, V, W exactly decompose T."""
    # Reshape U, V, W to 3D tensors
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need to compute sum_r U[:,r] * V[:,r] * W[:,r]
    # This is the tensor product of the columns.
    # We can use np.einsum or manual loop.
    # For speed, we can flatten and use matrix multiplication?
    # T_flatten = sum_r U[:,r] * V[:,r] * W[:,r]
    # This is not a simple matrix product.
    
    # Let's just loop over rank, it's small (18 or 36).
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Check if T_recon matches T
    if not np.allclose(T_recon, T, atol=tol, rtol=0):
        return False
    return True

def _solve_rank_9_233():
    """Try to find a rank 9 decomposition for M_{2,3,3}."""
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    rank = 9
    
    # Variables: U (6x9), V (9x9), W (6x9)
    # Total 54 + 81 + 54 = 189 variables
    
    # We will use scipy.optimize.least_squares
    from scipy.optimize import least_squares
    
    def residuals(x):
        U = x[:54].reshape(6, 9)
        V = x[54:135].reshape(9, 9)
        W = x[135:189].reshape(6, 9)
        
        T_recon = np.zeros_like(T)
        for r in range(9):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        
        return (T_recon - T).flatten()
    
    # Try a few random initializations
    np.random.seed(42)
    for _ in range(10):
        x0 = np.random.randint(-2, 3, size=189)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            if result.cost < 1e-10:
                x = result.x
                U = x[:54].reshape(6, 9)
                V = x[54:135].reshape(9, 9)
                W = x[135:189].reshape(6, 9)
                
                # Round to nearest integer or half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
                    return U_r, V_r, W_r
        except Exception:
            pass
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find rank 18 decomposition using rank 9 for M_{2,3,3}
    start_time = time.time()
    U_233, V_233, W_233 = _solve_rank_9_233()
    
    if U_233 is not None and time.time() - start_time < 100:
        # Construct rank 18 decomposition for M_{2,3,6}
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First 9 columns: Block 1
        U[:, :9] = U_233
        V[:, :9] = V_233
        # W for block 1: k in 0..2
        # W_233 has shape (6, 9) corresponding to k in 0..2, i in 0..1
        # W index is k*n + i.
        # For M_{2,3,6}, W index is k*n + i with k in 0..5.
        # So for k in 0..2, we use W_233.
        W[:, :9] = W_233
        
        # Next 9 columns: Block 2
        U[:, 9:] = U_233
        V[:, 9:] = V_233
        # W for block 2: k in 3..5
        # W index is k*n + i.
        # k=3, i=0 -> 3*2+0 = 6
        # k=3, i=1 -> 3*2+1 = 7
        # k=4, i=0 -> 4*2+0 = 8
        # k=4, i=1 -> 4*2+1 = 9
        # k=5, i=0 -> 5*2+0 = 10
        # k=5, i=1 -> 5*2+1 = 11
        # W_233 indices are 0..5.
        # We need to map W_233 to W[6:12, 9:]
        W[6:12, 9:] = W_233
        
        # Verify the full decomposition
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 36
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
