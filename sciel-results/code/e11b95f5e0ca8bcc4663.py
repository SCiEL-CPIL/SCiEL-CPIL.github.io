"""Matrix multiplication 3x3x3: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]


def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check the "on" entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > tol:
                    return False
    
    # Check "off" entries: T[i*m+j, j2*p+k2, k2*n+i2] = 0 when not matching
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i,j,k) != (i2,j2,k2):
                                val = np.sum(U[i*m+j, :] * V[j2*p+k2, :] * W[k2*n+i2, :])
                                if abs(val) > tol:
                                    return False
    
    return True


def _schoolbook(n=3, m=3, p=3):
    """Return the trivial rank-27 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _try_als_rank23(n=3, m=3, p=3, max_iters=200, n_restarts=10, tol=1e-8, time_limit=110.0):
    """
    Attempt to find a rank-23 decomposition via Alternating Least Squares (ALS).
    
    The idea: the matrix multiplication tensor T has 27 nonzero entries (at positions
    (i*m+j, j*p+k, k*n+i) for i,j,k in {0,1,2}). We want to find rank-1 tensors
    (u_r, v_r, w_r) such that sum_r u_r ⊗ v_r ⊗ w_r = T.
    
    We use ALS: fix two factors, solve for the third via least squares.
    """
    start_time = time.time()
    target_rank = 23
    
    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Flatten T to 1D for convenience
    T_flat = T.flatten()  # length 9*9*9 = 729
    
    # Precompute the index mapping
    # For a given (i, j, k, i2, j2, k2), the tensor entry T[i*m+j, j2*p+k2, k2*n+i2]
    # We need to check which entries are nonzero in T.
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize randomly
        np.random.seed(restart * 12345 + 42)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each row a, solve for U[a, :]
            # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a, :, :] = U[a, :] @ (V * W) where (V*W)[r, b, c] = V[b,r]*W[c,r]
            # Reshape: for each a, we have a system of (m*p)*(p*n) equations
            # T[a, b, c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_a be a matrix of shape (m*p*p*n, rank) where M_a[b*p*n+c, r] = V[b,r]*W[c,r]
            # Then T[a].flatten() = M_a @ U[a, :]
            
            # Build M (same for all a, just depends on V and W)
            # M has shape (m*p * p*n, rank)
            M = np.zeros((m*p * p*n, target_rank), dtype=float)
            for b in range(m*p):
                for c in range(p*n):
                    for r in range(target_rank):
                        M[b * p*n + c, r] = V[b, r] * W[c, r]
            
            # Solve for each row of U
            for a in range(n*m):
                rhs = T[a, :, :].flatten()
                # U[a, :] = M^+ rhs
                if np.linalg.matrix_rank(M) >= target_rank:
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                else:
                    U[a, :] = np.linalg.pinv(M) @ rhs
            
            # Update V similarly
            Mv = np.zeros((n*m * p*n, target_rank), dtype=float)
            for a in range(n*m):
                for c in range(p*n):
                    for r in range(target_rank):
                        Mv[a * p*n + c, r] = U[a, r] * W[c, r]
            
            for b in range(m*p):
                rhs = T[:, b, :].flatten()
                if np.linalg.matrix_rank(Mv) >= target_rank:
                    V[b, :] = np.linalg.lstsq(Mv, rhs, rcond=None)[0]
                else:
                    V[b, :] = np.linalg.pinv(Mv) @ rhs
            
            # Update W similarly
            Mw = np.zeros((n*m * m*p, target_rank), dtype=float)
            for a in range(n*m):
                for b in range(m*p):
                    for r in range(target_rank):
                        Mw[a * m*p + b, r] = U[a, r] * V[b, r]
            
            for c in range(p*n):
                rhs = T[:, :, c].flatten()
                if np.linalg.matrix_rank(Mw) >= target_rank:
                    W[c, :] = np.linalg.lstsq(Mw, rhs, rcond=None)[0]
                else:
                    W[c, :] = np.linalg.pinv(Mw) @ rhs
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            if err < tol:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return best_U, best_V, best_W, target_rank
        
        # Track best
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_err < tol:
        return best_U, best_V, best_W, target_rank
    
    return None


def _try_structural_rank23(n=3, m=3, p=3, time_limit=100.0):
    """
    Try a structural approach: start from rank-27 and try to merge 4 pairs of terms
    using linear combinations.
    
    The key idea: if we have two rank-1 terms (u1,v1,w1) and (u2,v2,w2), we can
    replace them with a single rank-1 term if there exists a scalar alpha such that
    the combined contribution equals the original.
    
    More practically: we look for 4 linear dependencies among the 27 tensor entries
    that allow us to express 4 of the rank-1 components as combinations of others.
    """
    start_time = time.time()
    
    # This is a heuristic search. For 3x3x3, the known rank-23 decompositions
    # use specific linear combinations. Let's try a few known patterns.
    
    # One known approach: the "Strassen-like" reduction for 3x3.
    # We can try to find 4 linear relations.
    
    # For now, this is a placeholder that will fall back to ALS or schoolbook.
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try ALS-based rank-23 search
    result = _try_als_rank23(n, m, p, max_iters=150, n_restarts=8, tol=1e-8, 
                              time_limit=deadline - start_time - 5)
    
    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try structural approach
    if time.time() < deadline - 5:
        result = _try_structural_rank23(n, m, p, time_limit=deadline - start_time - 5)
        if result is not None:
            U, V, W, rank = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-27
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
