import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _make_schoolbook(rank):
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=4, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # Vectorized dot product for speed
                # s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                # But U, V, W are already snapped, so float precision is fine.
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_optimize(rank, n=3, m=3, p=4, max_iters=500, seed=42, deadline=None):
    rng = np.random.RandomState(seed)
    
    # Initialize
    # Scale to keep values reasonable
    scale = 1.0 / np.sqrt(rank)
    U = rng.randn(n * m, rank) * scale
    V = rng.randn(m * p, rank) * scale
    W = rng.randn(p * n, rank) * scale
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # M is p x rank
                M_mat = np.zeros((p, rank))
                for k in range(p):
                    M_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M * U_ij^T = 1
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M_mat, np.ones(p), rcond=None)[0]
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # M is n x rank
                M_mat = np.zeros((n, rank))
                for i in range(n):
                    M_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Solve M * V_jk^T = 1
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M_mat, np.ones(n), rcond=None)[0]
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # M is m x rank
                M_mat = np.zeros((m, rank))
                for j in range(m):
                    M_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve M * W_ki^T = 1
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M_mat, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if err < 1e-10:
            break
            
    # Snap to half-integers
    if best_U is not None:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        # Verify snapped solution
        if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
            return U_snap, V_snap, W_snap
        else:
            # Try snapping with slightly different tolerance or just return if very close?
            # The problem requires EXACT decomposition.
            # If snapping fails, this rank/seed combination is a failure.
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Fallback
    best_result = _make_schoolbook(schoolbook_rank)
    
    # Try to find lower rank
    # Start from 35 down to 12
    # We prioritize higher ranks (closer to schoolbook) as they are more likely to exist
    for rank in range(schoolbook_rank - 1, 11, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few seeds for each rank
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            result = _als_optimize(rank, n, m, p, max_iters=200, seed=seed, deadline=deadline - 10)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
                
        if found:
            break
            
    return best_result
