"""Matrix multiplication 4x4 by 4x5 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# Deadline guard
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_remaining():
    # Keep 15s safety margin before the 150s kill
    return 135.0 - _elapsed()

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product term.
    The tensor has shape (n*m, m*p, p*n) in the factor index convention:
    U index: i*m + j (row of A, col of A)
    V index: j*p + k (row of B, col of B)
    W index: k*n + i (row of C, col of C)
    
    T[a, b, c] = 1 iff there exist i,j,k with:
      a = i*m + j
      b = j*p + k
      c = k*n + i
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Compute reconstructed tensor
    # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,c]
    # Efficient: for each r, outer product
    recon = np.zeros_like(T)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(recon, T, atol=tol)

def _round_to_halfint(M):
    """Round matrix entries to nearest half-integer (multiples of 0.5)."""
    return np.round(M * 2.0) / 2.0

def _als_step(U, V, W, T, n, m, p, lr=0.01, iters=50):
    """One ALS refinement step: update U, V, W alternatively to minimize ||recon - T||.
    Returns updated U, V, W.
    """
    rank = U.shape[1]
    # Precompute contractions for efficiency
    # For updating U[:,r]: minimize sum_{b,c} (U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
    # = sum_{b,c} U[a,r]^2 * V[b,r]^2 * W[c,r]^2 - 2*U[a,r]*V[b,r]*W[c,r]*T[a,b,c] + const
    # Optimal U[a,r] = sum_{b,c} V[b,r]*W[c,r]*T[a,b,c] / sum_{b,c} V[b,r]^2*W[c,r]^2
    
    for it in range(iters):
        # Update U
        for r in range(rank):
            # numerator[a] = sum_{b,c} V[b,r]*W[c,r]*T[a,b,c]
            # denominator = sum_{b,c} V[b,r]^2 * W[c,r]^2
            num = np.einsum('b,c,abc->a', V[:, r], W[:, r], T)
            den = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
            if den > 1e-12:
                U[:, r] = num / den
            # Gradient step blending
            U[:, r] = 0.9 * U[:, r] + 0.1 * U[:, r]  # keep as is for now
        
        # Update V
        for r in range(rank):
            num = np.einsum('a,c,abc->b', U[:, r], W[:, r], T)
            den = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
            if den > 1e-12:
                V[:, r] = num / den
        
        # Update W
        for r in range(rank):
            num = np.einsum('a,b,abc->c', U[:, r], V[:, r], T)
            den = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
            if den > 1e-12:
                W[:, r] = num / den
    
    return U, V, W

def _compute_error(U, V, W, T):
    """Compute Frobenius error of the decomposition."""
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(recon - T)

def _try_rank(rank, n, m, p, T, max_time=30.0):
    """Try to find a rank decomposition by ALS from random init.
    Returns (U, V, W) if successful, else None.
    """
    time_limit = max_time
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if _deadline_remaining() < 10.0:
            break
        U, V, W = _als_step(U, V, W, T, n, m, p, iters=10)
        err = _compute_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try rounding and checking
        if err < 1e-6:
            U_r = _round_to_halfint(best_U)
            V_r = _round_to_halfint(best_V)
            W_r = _round_to_halfint(best_W)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                return U_r, V_r, W_r
        
        # If error is very small, try rounding
        if err < 0.1:
            U_r = _round_to_halfint(U)
            V_r = _round_to_halfint(V)
            W_r = _round_to_halfint(W)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                return U_r, V_r, W_r
    
    return None

def _strassen_init(n, m, p):
    """Create a Strassen-like initial decomposition.
    For 4x4 * 4x5, we can use block structure.
    Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2 (with padding for 5).
    Actually, let's just use the schoolbook as baseline and try to reduce.
    """
    rank = n * m * p  # 80
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _gaussian_elimination_reduce(U, V, W, T, n, m, p):
    """Try to reduce rank by finding linear dependencies among columns.
    Check if any column is a linear combination of others.
    """
    rank = U.shape[1]
    # Build the tensor and check for dependencies
    # For each pair of columns, check if one is redundant
    # This is complex; skip for now
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook (rank 80)
    best_rank = n * m * p
    U, V, W = _strassen_init(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various ranks from low to high
    # Known lower bound for 4x4*4x5: at least max(n*m, m*p, p*n) = max(16,20,20) = 20
    # But realistically, for 4x4*4x4 it's known to be at most 49 (Lader) and at least 27.
    # For 4x4*4x5, let's try ranks from 20 upward.
    
    # First, try to verify schoolbook
    if _verify(U, V, W, n, m, p):
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try reducing rank via ALS from various initializations
    for target_rank in range(20, best_rank + 1):
        if _deadline_remaining() < 15.0:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 3 if target_rank > 30 else 5
        found = False
        
        for restart in range(max_restarts):
            if _deadline_remaining() < 15.0:
                break
            
            # Initialize with random values
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Run ALS
            U_c, V_c, W_c = U_init.copy(), V_init.copy(), W_init.copy()
            best_err_local = float('inf')
            
            for it in range(100):
                if _deadline_remaining() < 15.0:
                    break
                
                # Update U
                for r in range(target_rank):
                    num = np.einsum('b,c,abc->a', V_c[:, r], W_c[:, r], T)
                    den = np.sum(V_c[:, r]**2) * np.sum(W_c[:, r]**2)
                    if den > 1e-12:
                        U_c[:, r] = num / den
                
                # Update V
                for r in range(target_rank):
                    num = np.einsum('a,c,abc->b', U_c[:, r], W_c[:, r], T)
                    den = np.sum(U_c[:, r]**2) * np.sum(W_c[:, r]**2)
                    if den > 1e-12:
                        V_c[:, r] = num / den
                
                # Update W
                for r in range(target_rank):
                    num = np.einsum('a,b,abc->c', U_c[:, r], V_c[:, r], T)
                    den = np.sum(U_c[:, r]**2) * np.sum(V_c[:, r]**2)
                    if den > 1e-12:
                        W_c[:, r] = num / den
                
                err = _compute_error(U_c, V_c, W_c, T)
                if err < best_err_local:
                    best_err_local = err
                
                if err < 1e-8:
                    # Try rounding
                    U_r = _round_to_halfint(U_c)
                    V_r = _round_to_halfint(V_c)
                    W_r = _round_to_halfint(W_c)
                    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        found = True
                        break
                
                if err < 0.5:
                    U_r = _round_to_halfint(U_c)
                    V_r = _round_to_halfint(V_c)
                    W_r = _round_to_halfint(W_c)
                    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        found = True
                        break
            
            if found:
                break
        
        if found:
            # Try to reduce further
            continue
        else:
            # This rank didn't work, try higher
            continue
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
