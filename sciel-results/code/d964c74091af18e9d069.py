import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, rank):
    """Verify if U, V, W exactly decompose the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iters=100, tol=1e-6):
    """
    Attempt to find a rank-R decomposition using ALS.
    Returns (U, V, W) as lists of lists if successful, else None.
    """
    # Initialize with random small integers
    U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
    V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
    W = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
    
    # Ensure no zero columns
    for c in range(rank):
        if np.all(U[:, c] == 0): U[np.random.randint(n*m), c] = 1.0
        if np.all(V[:, c] == 0): V[np.random.randint(m*p), c] = 1.0
        if np.all(W[:, c] == 0): W[np.random.randint(p*n), c] = 1.0

    prev_err = float('inf')
    for it in range(max_iters):
        # Solve for U
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # For fixed i, j: T[i,j,:] = U[i,j,:] * (V[:,j,:] * W[:,i,:]) ... wait, indices are tricky.
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We can vectorize the updates.
        # Update U:
        # For each i, j, we want to find U[i*m+j, :] such that
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
        # This is a linear system: A_ik @ u = 1
        # Where A_ik[c, k'] = V[j*p+k', c] * W[k'*n+i, c]
        
        # This is still O(n*m*p*rank^2) per iteration.
        # With n=2, m=3, p=4, rank=20, this is fast.
        
        for i in range(n):
            for j in range(m):
                # Construct matrix A (rank x p)
                A = np.zeros((rank, p))
                for k in range(p):
                    A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ u = ones(p)
                try:
                    u, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except np.linalg.LinAlgError:
                    pass
        
        # Update V:
        for j in range(m):
            for k in range(p):
                # Construct matrix B (rank x n)
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed j, k: sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c] = 1 for all i
                B = np.zeros((rank, n))
                for i in range(n):
                    B[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except np.linalg.LinAlgError:
                    pass
        
        # Update W:
        for k in range(p):
            for i in range(n):
                # Construct matrix C (rank x m)
                # For fixed k, i: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1 for all j
                C = np.zeros((rank, m))
                for j in range(m):
                    C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    pass
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, rank):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try ranks from 19 upwards (known optimal is 19)
    for rank in range(19, 24):
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        num_trials = 50 if rank == 19 else 100
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
            
            result = _als_decompose(n, m, p, rank, max_iters=200, tol=1e-8)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
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
