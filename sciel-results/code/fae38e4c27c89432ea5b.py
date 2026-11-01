"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build the multiplication tensor T with shape (n*m, m*p, n*p)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T"""
        T_recon = np.einsum('ac,bc,cc->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def try_rank(r, max_attempts=30, max_iters=300):
        """Try to find rank-r decomposition using ALS"""
        if time.time() > deadline - 10:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(n*p, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: T[a,b,c] = sum_c' U[a,c'] * V[b,c'] * W[c,c']
                # For fixed a: T[a,:,:] = sum_c' U[a,c'] * outer(V[:,c'], W[:,c'])
                P = np.zeros((r, m*p * n*p))
                for cp in range(r):
                    P[cp, :] = np.outer(V[:, cp], W[:, cp]).flatten()
                T_flat = T.reshape(n*m, m*p * n*p)
                try:
                    PPt = P @ P.T
                    U = T_flat @ P.T @ np.linalg.inv(PPt)
                except:
                    break
                
                # Update V
                PW = np.zeros((r, n*m * n*p))
                for cp in range(r):
                    PW[cp, :] = np.outer(U[:, cp], W[:, cp]).flatten()
                T_flat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * n*p)
                try:
                    PWt = PW @ PW.T
                    V = T_flat2 @ PW.T @ np.linalg.inv(PWt)
                except:
                    break
                
                # Update W
                PU = np.zeros((r, n*m * m*p))
                for cp in range(r):
                    PU[cp, :] = np.outer(U[:, cp], V[:, cp]).flatten()
                T_flat3 = T.transpose(2, 0, 1).reshape(n*p, n*m * m*p)
                try:
                    PUt = PU @ PU.T
                    W = T_flat3 @ PU.T @ np.linalg.inv(PUt)
                except:
                    break
                
                if it % 20 == 0:
                    if check_decomposition(U, V, W):
                        return (U, V, W)
            
            if check_decomposition(U, V, W):
                return (U, V, W)
        
        return None
    
    # Try increasing ranks
    for r in range(1, 15):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
