import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _reconstruct_error(U, V, W, n, m, p, r):
    """Compute Frobenius norm of reconstruction error for rank-r decomposition"""
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    C = np.einsum('ij l, jk l, ki l -> ijk', U_t, V_t, W_t)
    return np.sqrt(np.sum((C - 1.0) ** 2))

def _als_decompose(n, m, p, r, max_iter=1000, tol=1e-12, time_budget=10.0):
    """
    Vectorized Alternating Least Squares.
    """
    start_time = time.time()
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Update U
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # A[i, j, k, l] = V[j, k, l] * W[k, i, l]
        A = np.einsum('jkl, kil -> i j k l', V_t, W_t)
        
        # M[i, j, l1, l2] = sum_k A[i, j, k, l1] * A[i, j, k, l2]
        M = np.einsum('ij k l1, ij k l2 -> i j l1 l2', A, A)
        
        # b[i, j, l1] = sum_k A[i, j, k, l1]
        b = np.einsum('ij k l1 -> i j l1', A)
        
        for i in range(n):
            for j in range(m):
                try:
                    Mi = M[i, j] + 1e-8 * np.eye(r)
                    bi = b[i, j]
                    u, _, _, _ = np.linalg.lstsq(Mi, bi, rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass

        # Update V
        U_t = U.reshape(n, m, r)
        W_t = W.reshape(p, n, r)
        
        # A_V[i, j, k, l] = U[i, j, l] * W[k, i, l]
        A_V = np.einsum('ij l, ki l -> i j k l', U_t, W_t)
        
        # M_V[j, k, l1, l2] = sum_i A_V[i, j, k, l1] * A_V[i, j, k, l2]
        M_V = np.einsum('i j k l1, i j k l2 -> j k l1 l2', A_V, A_V)
        
        # b_V[j, k, l1] = sum_i A_V[i, j, k, l1]
        b_V = np.einsum('i j k l1 -> j k l1', A_V)
        
        for j in range(m):
            for k in range(p):
                try:
                    Mj = M_V[j, k] + 1e-8 * np.eye(r)
                    bj = b_V[j, k]
                    v, _, _, _ = np.linalg.lstsq(Mj, bj, rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass

        # Update W
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        
        # A_W[i, j, k, l] = U[i, j, l] * V[j, k, l]
        A_W = np.einsum('ij l, jk l -> i j k l', U_t, V_t)
        
        # M_W[k, i, l1, l2] = sum_j A_W[i, j, k, l1] * A_W[i, j, k, l2]
        M_W = np.einsum('i j k l1, i j k l2 -> k i l1 l2', A_W, A_W)
        
        # b_W[k, i, l1] = sum_j A_W[i, j, k, l1]
        b_W = np.einsum('i j k l1 -> k i l1', A_W)
        
        for k in range(p):
            for i in range(n):
                try:
                    Mk = M_W[k, i] + 1e-8 * np.eye(r)
                    bk = b_W[k, i]
                    w, _, _, _ = np.linalg.lstsq(Mk, bk, rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass

        # Compute error
        error = _reconstruct_error(U, V, W, n, m, p, r)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return U, V, W, error

def _snap_to_half_int(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough. Returns (snapped, success)."""
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped, True
    return arr, False

def _verify_exact(U, V, W, n, m, p, r, tol=1e-9):
    """Verify that the decomposition is exact (within tolerance)."""
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    C = np.einsum('ij l, jk l, ki l -> ijk', U_t, V_t, W_t)
    if not np.allclose(C, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    fallback = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    
    # Search from rank 18 up to schoolbook
    best_result = fallback
    best_rank = rank_sb
    
    for r in range(18, rank_sb + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            try:
                U, V, W, error = _als_decompose(n, m, p, r, max_iter=200, tol=1e-12, time_budget=2.0)
            except:
                continue
            
            if error < 1e-6:
                # Snap to half-integers
                U_snap, ok1 = _snap_to_half_int(U, tol=1e-5)
                V_snap, ok2 = _snap_to_half_int(V, tol=1e-5)
                W_snap, ok3 = _snap_to_half_int(W, tol=1e-5)
                
                if ok1 and ok2 and ok3:
                    # Verify exact
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, r, tol=1e-9):
                        result = {
                            "U": [[float(x) for x in row] for row in U_snap],
                            "V": [[float(x) for x in row] for row in V_snap],
                            "W": [[float(x) for x in row] for row in W_snap]
                        }
                        if r < best_rank:
                            best_rank = r
                            best_result = result
                        return best_result
        
        # If we found a good solution, we can stop
        if best_rank < rank_sb:
            break
    
    return best_result
