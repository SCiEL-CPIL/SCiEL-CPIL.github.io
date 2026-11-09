"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array.
    
    T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    T[a,b,c] = 1 if j=j' and k'=k'' and i''=i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries (should be 0)
    # T[i*m+j, jp*p+kp, kp*n+ip] = 0 unless j=jp, kp=k, ip=i
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for ip in range(n):
                        if jp == j and kp == k and ip == i:
                            continue
                        val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kp*n+ip, :])
                        if abs(val) > tol:
                            return False
    return True


def _als_decomposition(T, n, m, p, rank, max_iter=200, tol=1e-12, seed=42, polish=False):
    """Use ALS to find a rank-r decomposition of the tensor T."""
    rng = np.random.RandomState(seed)
    
    # Better initialization: use structured random with some zero patterns
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Add some structure: make first few columns have sparsity
    if rank > 10:
        U[:n*m//2, :rank//3] = 0
        V[:m*p//2, :rank//3] = 0
        W[:p*n//2, :rank//3] = 0
    
    for it in range(max_iter):
        # Precompute design matrices using Kronecker-like structure
        # D_U[b,c,r] = V[b,r]*W[c,r]
        D_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, rank)
        
        # D_V[a,c,r] = U[a,r]*W[c,r]
        D_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, rank)
        
        # D_W[a,b,r] = U[a,r]*V[b,r]
        D_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, rank)
        
        # Update U
        for a in range(n * m):
            target = T[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(D_U, target, rcond=None)[0]
        
        # Update V
        for b in range(m * p):
            target = T[:, b, :].flatten()
            V[b, :] = np.linalg.lstsq(D_V, target, rcond=None)[0]
        
        # Update W
        for c in range(p * n):
            target = T[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(D_W, target, rcond=None)[0]
        
        # Check convergence
        if it % 10 == 0 or it == max_iter - 1:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < tol:
                break
    
    # Polish pass: if close, do additional iterations with tighter tolerance
    if polish:
        for it in range(50):
            D_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, rank)
            D_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, rank)
            D_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, rank)
            
            for a in range(n * m):
                target = T[a, :, :].flatten()
                U[a, :] = np.linalg.lstsq(D_U, target, rcond=None)[0]
            
            for b in range(m * p):
                target = T[:, b, :].flatten()
                V[b, :] = np.linalg.lstsq(D_V, target, rcond=None)[0]
            
            for c in range(p * n):
                target = T[:, :, c].flatten()
                W[c, :] = np.linalg.lstsq(D_W, target, rcond=None)[0]
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < 1e-14:
                break
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank(T, n, m, p, rank, time_budget=10.0, seed=42, n_seeds=3):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    for s in range(n_seeds):
        remaining = time_budget - (time.time() - start)
        if remaining < 2.0:
            break
        try:
            polish = (s == n_seeds - 1)  # Only polish on last attempt
            U, V, W = _als_decomposition(T, n, m, p, rank, max_iter=150, tol=1e-12, seed=seed + s, polish=polish)
            if time.time() - start > time_budget:
                break
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                return U, V, W
        except Exception:
            pass
    return None


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 20.0:
        return best_result
    
    # Try ALS with various ranks, starting from moderate and going down
    # Schoolbook is 72. Try to find something much lower.
    # Use more seeds for lower ranks to escape local minima
    rank_schedule = [
        (40, 2), (35, 3), (30, 4), (28, 5), (26, 5), (24, 6), (22, 6), (20, 7)
    ]
    
    for rank, n_seeds in rank_schedule:
        remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
        if remaining < 20.0:
            break
        
        # Allocate time based on remaining budget and number of seeds
        time_budget = min(remaining * 0.3, 25.0)
        result = _try_rank(T, n, m, p, rank, time_budget=time_budget, seed=42 + rank, n_seeds=n_seeds)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = rank
    
    return best_result
