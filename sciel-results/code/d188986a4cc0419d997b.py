"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _snap_to_half(x):
    """Snap x to nearest half-integer."""
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
    """Fast verification assuming already numpy arrays."""
    if Ua.shape != (n*m, Ua.shape[1]) or Va.shape != (m*p, Va.shape[1]) or Wa.shape != (p*n, Wa.shape[1]):
        return False
    if Ua.shape[1] != Va.shape[1] or Ua.shape[1] != Wa.shape[1]:
        return False
    rank = Ua.shape[1]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=atol)

def _snap_numpy(arr):
    """Snap numpy array to nearest half-integer."""
    return np.round(2 * arr) / 2.0

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    """
    Try to find a valid decomposition of given rank using alternating linear solve.
    Uses numpy-vectorized matrix construction for speed.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with random values
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Update U: for each (i,j), solve linear system
        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                V_j = V[j*p:(j+1)*p, :]  # (p, rank)
                W_i = W[i::n, :]  # (p, rank)
                M = V_j * W_i
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve linear system
        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                U_j = U[j::m, :]  # (n, rank)
                W_k = W[k::p, :]  # (n, rank)
                M = U_j * W_k
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve linear system
        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                U_i = U[i*m:(i+1)*m, :]  # (m, rank)
                V_k = V[k::p, :]  # (m, rank)
                M = U_i * V_k
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        # Snap to half-integers
        Ua_snapped = _snap_numpy(best_U)
        Va_snapped = _snap_numpy(best_V)
        Wa_snapped = _snap_numpy(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        # Also try without snapping
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_gradient(n, m, p, rank, max_iter=3000, seed=None):
    """Try to find a valid decomposition of given rank using gradient descent with adaptive learning rate."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    lr = 0.05
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        error = np.max(np.abs(err))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if error < 1e-10:
            break
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.8
    
    Ua = best_U.reshape(n*m, rank)
    Va = best_V.reshape(m*p, rank)
    Wa = best_W.reshape(p*n, rank)
    
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_structure(n, m, p, rank, seed=None):
    """
    Try structured initialization based on matrix multiplication properties.
    Uses Kronecker-like patterns to exploit structure.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strategy 1: Block-structured initialization
    # Divide indices into blocks and use Kronecker-like patterns
    for r in range(rank):
        # Assign a "block" pattern to this rank component
        pattern = rng.choice(['sparse', 'dense', 'kronecker'])
        
        if pattern == 'sparse':
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            V[j*p+k, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            W[k*n+i, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            
        elif pattern == 'dense':
            # Fill multiple entries with small values
            for _ in range(rng.randint(2, 4)):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, r] += rng.uniform(-0.3, 0.3)
                V[j*p+k, r] += rng.uniform(-0.3, 0.3)
                W[k*n+i, r] += rng.uniform(-0.3, 0.3)
            
        else:  # kronecker
            # Use binary representation of indices for Kronecker structure
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            # Create a pattern based on bit patterns
            val = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            U[i*m+j, r] = val
            # Add correlated entries based on shared index bits
            if i > 0:
                U[(i-1)*m+j, r] += val * 0.5
            if j > 0:
                U[i*m+(j-1), r] += val * 0.5
            V[j*p+k, r] = val
            if j > 0:
                V[(j-1)*p+k, r] += val * 0.5
            if k > 0:
                V[j*p+(k-1), r] += val * 0.5
            W[k*n+i, r] = val
            if k > 0:
                W[(k-1)*n+i, r] += val * 0.5
            if i > 0:
                W[k*n+(i-1), r] += val * 0.5
    
    return U, V, W

def _try_rank_kronecker_init(n, m, p, rank, seed=None):
    """
    Kronecker product-based initialization for structured search.
    Exploits the fact that matmul can be decomposed into smaller sub-problems.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Try different Kronecker decompositions
    # For 3x5x6, we can think of it as combinations of smaller problems
    # e.g., (2+1)x(3+2)x(4+2) type decompositions
    
    # Method: Use random Kronecker-like structured matrices
    # Split rank into groups and assign Kronecker patterns
    n_groups = min(rank, 6)
    group_size = rank // n_groups
    if group_size < 1:
        group_size = 1
        n_groups = rank
    
    for g in range(n_groups):
        start_r = g * group_size
        end_r = min(start_r + group_size, rank)
        
        # Choose a small base problem size for this group
        base_n = rng.choice([1, 2, 3])
        base_m = rng.choice([1, 2, 3])
        base_p = rng.choice([1, 2, 3])
        
        # Create base Kronecker matrices
        base_U = np.zeros((base_n * base_m, 1))
        base_V = np.zeros((base_m * base_p, 1))
        base_W = np.zeros((base_p * base_n, 1))
        
        # Fill with structured values
        for bi in range(base_n):
            for bj in range(base_m):
                base_U[bi*base_m+bj, 0] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for bj in range(base_m):
            for bk in range(base_p):
                base_V[bj*base_p+bk, 0] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for bk in range(base_p):
            for bi in range(base_n):
                base_W[bk*base_n+bi, 0] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        
        # Map base indices to full indices
        for r in range(start_r, end_r):
            # Map each base index to a random full index
            for bi in range(base_n):
                for bj in range(base_m):
                    if base_U[bi*base_m+bj, 0] != 0:
                        # Map to a random (i,j) in the full problem
                        i = rng.randint(0, n-1)
                        j = rng.randint(0, m-1)
                        U[i*m+j, r] += base_U[bi*base_m+bj, 0] * (1.0 / max(1, group_size))
            
            for bj in range(base_m):
                for bk in range(base_p):
                    if base_V[bj*base_p+bk, 0] != 0:
                        j = rng.randint(0, m-1)
                        k = rng.randint(0, p-1)
                        V[j*p+k, r] += base_V[bj*base_p+bk, 0] * (1.0 / max(1, group_size))
            
            for bk in range(base_p):
                for bi in range(base_n):
                    if base_W[bk*base_n+bi, 0] != 0:
                        k = rng.randint(0, p-1)
                        i = rng.randint(0, n-1)
                        W[k*n+i, r] += base_W[bk*base_n+bi, 0] * (1.0 / max(1, group_size))
    
    return U, V, W

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    """High-precision polish pass using alternating linear solve with numpy-vectorized ops."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    # Precompute index arrays for vectorization
    idx_i = np.arange(n)
    idx_j = np.arange(m)
    idx_k = np.arange(p)
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        # Update U: for each (i,j), solve M @ U_ij = ones(p)
        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                # Vectorized construction of M
                V_j = Va[j*p:(j+1)*p, :]  # (p, rank)
                W_i = Wa[i::n, :]  # (p, rank) - stride n
                M = V_j * W_i  # (p, rank)
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_j = Ua[j::m, :]  # (n, rank) - stride m
                W_k = Wa[k::p, :]  # (n, rank) - stride p
                M = U_j * W_k  # (n, rank)
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_i = Ua[i*m:(i+1)*m, :]  # (m, rank)
                V_k = Va[k::p, :]  # (m, rank) - stride p
                M = U_i * V_k  # (m, rank)
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_rank_altsolve_vectorized(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    """
    Vectorized alternating linear solve - much faster than loop-based version.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute index arrays
    idx_i = np.arange(n)
    idx_j = np.arange(m)
    idx_k = np.arange(p)
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        # Vectorized U update: for each (i,j), solve M_ij @ U_ij = ones(p)
        # M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                V_j = V[j*p:(j+1)*p, :]  # (p, rank)
                W_i = W[i::n, :]  # (p, rank)
                M = V_j * W_i  # (p, rank)
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Vectorized V update: for each (j,k), solve M_jk @ V_jk = ones(n)
        # M_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                U_j = U[j::m, :]  # (n, rank)
                W_k = W[k::p, :]  # (n, rank)
                M = U_j * W_k  # (n, rank)
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Vectorized W update: for each (k,i), solve M_ki @ W_ki = ones(m)
        # M_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                U_i = U[i*m:(i+1)*m, :]  # (m, rank)
                V_k = V[k::p, :]  # (m, rank)
                M = U_i * V_k  # (m, rank)
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        Ua_snapped = _snap_numpy(best_U)
        Va_snapped = _snap_numpy(best_V)
        Wa_snapped = _snap_numpy(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    """
    Try to find a valid decomposition of given rank using alternating linear solve.
    Uses numpy-vectorized matrix construction for speed.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with random values
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Update U: for each (i,j), solve linear system
        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                V_j = V[j*p:(j+1)*p, :]  # (p, rank)
                W_i = W[i::n, :]  # (p, rank)
                M = V_j * W_i
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve linear system
        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                U_j = U[j::m, :]  # (n, rank)
                W_k = W[k::p, :]  # (n, rank)
                M = U_j * W_k
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve linear system
        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                U_i = U[i*m:(i+1)*m, :]  # (m, rank)
                V_k = V[k::p, :]  # (m, rank)
                M = U_i * V_k
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        # Snap to half-integers
        Ua_snapped = _snap_numpy(best_U)
        Va_snapped = _snap_numpy(best_V)
        Wa_snapped = _snap_numpy(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        # Also try without snapping
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Lower bound for rank: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    min_rank = max(n*m, m*p, n*p)
    
    # Try ranks from min_rank upward
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 15.0:
            break
        
        found = False
        
        # Try alternating linear solve with multiple seeds - increased attempts
        for attempt in range(12):
            if _remaining() < 8.0:
                break
            
            result = _try_rank_altsolve(n, m, p, rank, max_iter=200, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    # Try to polish
                    Ua, Va, Wa = _polish(U, V, W, n, m, p, max_iter=30)
                    U_snapped = _snap_numpy(Ua).tolist()
                    V_snapped = _snap_numpy(Va).tolist()
                    W_snapped = _snap_numpy(Wa).tolist()
                    
                    if _verify_fast(U_snapped, V_snapped, W_snapped, n, m, p):
                        best = {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    else:
                        best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        # Try gradient descent if alternating solve didn't work
        if not found:
            for attempt in range(5):
                if _remaining() < 8.0:
                    break
                
                result = _try_rank_gradient(n, m, p, rank, max_iter=1500, seed=rank*100+100+attempt)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        found = True
                        break
        
        # Try structured initialization
        if not found:
            for attempt in range(4):
                if _remaining() < 5.0:
                    break
                
                U_init, V_init, W_init = _try_rank_structure(n, m, p, rank, seed=rank*1000+attempt)
                # Run alternating solve from this initialization using _polish
                Ua, Va, Wa = _polish(U_init, V_init, W_init, n, m, p, max_iter=150, tol=1e-10)
                
                U3 = Ua.reshape(n, m, rank)
                V3 = Va.reshape(m, p, rank)
                W3 = Wa.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                error = np.max(np.abs(T - 1.0))
                
                if error < 1e-10:
                    U_out = _snap_numpy(Ua).tolist()
                    V_out = _snap_numpy(Va).tolist()
                    W_out = _snap_numpy(Wa).tolist()
                    
                    if _verify_fast(U_out, V_out, W_out, n, m, p):
                        best = {"U": U_out, "V": V_out, "W": W_out}
                        best_rank = rank
                        found = True
                        break
        
        # Try Kronecker-structured initialization
        if not found:
            for attempt in range(4):
                if _remaining() < 5.0:
                    break
                
                U_init, V_init, W_init = _try_rank_kronecker_init(n, m, p, rank, seed=rank*2000+attempt)
                Ua, Va, Wa = _polish(U_init, V_init, W_init, n, m, p, max_iter=150, tol=1e-10)
                
                U3 = Ua.reshape(n, m, rank)
                V3 = Va.reshape(m, p, rank)
                W3 = Wa.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                error = np.max(np.abs(T - 1.0))
                
                if error < 1e-10:
                    U_out = _snap_numpy(Ua).tolist()
                    V_out = _snap_numpy(Va).tolist()
                    W_out = _snap_numpy(Wa).tolist()
                    
                    if _verify_fast(U_out, V_out, W_out, n, m, p):
                        best = {"U": U_out, "V": V_out, "W": W_out}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    # Final check: ensure the best result is valid
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
