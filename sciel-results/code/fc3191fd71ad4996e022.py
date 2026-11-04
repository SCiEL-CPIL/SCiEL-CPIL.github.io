"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x7."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decomposition
    # The tensor has 63 non-zero entries. Let's try ranks from low to high.
    
    def attempt_rank(rank, n_iter=50, seed=42):
        """Attempt rank-r decomposition via alternating least squares."""
        np.random.seed(seed)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank) 
        W = np.random.randn(pn, rank)
        
        for it in range(n_iter):
            # Update U
            Q = (V[:, :, None] * W[None, :, :]).reshape(mp, pn, rank).transpose(2, 0, 1).reshape(rank, mp*pn)
            T_flat = T.reshape(nm, mp*pn)
            QQT = Q @ Q.T + 1e-10 * np.eye(rank)
            U = T_flat @ Q.T @ np.linalg.solve(QQT, np.eye(rank))
            
            # Update V
            P = (U[:, :, None] * W[None, :, :]).reshape(nm, pn, rank).transpose(2, 0, 1).reshape(rank, nm*pn)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            PPT = P @ P.T + 1e-10 * np.eye(rank)
            V = T_flat_v @ P.T @ np.linalg.solve(PPT, np.eye(rank))
            
            # Update W
            R = (U[:, :, None] * V[None, :, :]).reshape(nm, mp, rank).transpose(2, 0, 1).reshape(rank, nm*mp)
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm*mp)
            RRT = R @ R.T + 1e-10 * np.eye(rank)
            W = T_flat_w @ R.T @ np.linalg.solve(RRT, np.eye(rank))
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.abs(T - T_recon).max()
        return U, V, W, err
    
    # Try various ranks
    best_result = None
    for rank in range(25, 64):
        if time.time() > deadline - 20:
            break
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 10:
                break
            U, V, W, err = attempt_rank(rank, n_iter=30, seed=seed)
            if err < 1e-6:
                best_result = (rank, U, V, W)
                break
        if best_result:
            break
    
    if best_result:
        rank, U, V, W = best_result
        return {
            "U": [[float(x) for x in U[r]] for r in range(nm)],
            "V": [[float(x) for x in V[r]] for r in range(mp)],
            "W": [[float(x) for x in W[r]] for r in range(pn)]
        }
    
    # Fallback: schoolbook
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
