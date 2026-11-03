import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    
    # Try a few ranks
    for rank in [100, 110, 120]:
        if time.time() > deadline:
            break
            
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Precompute some things
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n*m):
                # A[a*(p*n)+b, c] = V[a,c] * W[b,c]
                A = np.zeros((m*p*p*n, rank))
                for a in range(m*p):
                    for b in range(p*n):
                        A[a*(p*n)+b, :] = V[a, :] * W[b, :]
                U[i, :] = np.linalg.lstsq(A, T[i].flatten(), rcond=None)[0]
            
            # Update V
            for a in range(m*p):
                A = np.zeros((n*m*p*n, rank))
                for i in range(n*m):
                    for b in range(p*n):
                        A[i*(p*n)+b, :] = U[i, :] * W[b, :]
                V[a, :] = np.linalg.lstsq(A, T[:, a, :].reshape(-1), rcond=None)[0]
            
            # Update W
            for b in range(p*n):
                A = np.zeros((n*m*m*p, rank))
                for i in range(n*m):
                    for a in range(m*p):
                        A[i*(m*p)+a, :] = U[i, :] * V[a, :]
                W[b, :] = np.linalg.lstsq(A, T[:, :, b].reshape(-1), rcond=None)[0]
        
        recon = np.einsum('ic,ac,bc->iab', U, V, W)
        residual = np.linalg.norm(recon - T)
        
        if residual < 1e-5:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            recon_rounded = np.einsum('ic,ac,bc->iab', U_rounded, V_rounded, W_rounded)
            if np.allclose(recon_rounded, T, atol=1e-9):
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break
    
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
