"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 130
    
    # Build tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomp(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(recon, T, atol=1e-10)
    
    # Try ALS for various ranks
    def als(T, rank, seed=0, max_iters=500, tol=1e-14):
        dim1, dim2, dim3 = T.shape
        np.random.seed(seed)
        U = np.random.randn(dim1, rank) * 0.1
        V = np.random.randn(dim2, rank) * 0.1
        W = np.random.randn(dim3, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            VW = np.kron(V, W)
            for i in range(dim1):
                U[i] = np.linalg.lstsq(VW, T[i].ravel(), rcond=None)[0]
            
            # Update V
            UW = np.kron(U, W)
            for j in range(dim2):
                V[j] = np.linalg.lstsq(UW, T[:, j, :].ravel(), rcond=None)[0]
            
            # Update W
            UV = np.kron(U, V)
            for k in range(dim3):
                W[k] = np.linalg.lstsq(UV, T[:, :, k].ravel(), rcond=None)[0]
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(recon - T) / max(np.linalg.norm(T), 1e-15)
            if err < tol:
                break
        
        return U, V, W, err
    
    best = None
    for rank in range(32, 65):
        if time.time() > deadline - 25:
            break
        for seed in range(3):
            if time.time() > deadline - 25:
                break
            U, V, W, err = als(T, rank, seed=seed, max_iters=300)
            if err < 1e-10:
                best = (rank, U, V, W)
                break
        if best:
            break
    
    if best is None:
        # Schoolbook fallback
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best = (rank, U, V, W)
    
    rank, U, V, W = best
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
