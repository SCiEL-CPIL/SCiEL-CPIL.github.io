import time
import numpy as np
import scipy.optimize
from itertools import product

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor for n x m * m x p."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _snap_to_half_integers(U, V, W, tol=1e-8):
    """Try to snap factor matrices to half-integer values while preserving validity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try different rounding strategies
    for mult in [2, 4, 8]:
        U_r = np.round(U * mult) / mult
        V_r = np.round(V * mult) / mult
        W_r = np.round(W * mult) / mult
        T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.max(np.abs(T_recon - T)) < tol:
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try local search: adjust individual entries
    best_err = np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
    for _ in range(2000):
        # Pick a random entry to adjust
        mat = np.random.choice([U, V, W])
        row = np.random.randint(0, mat.shape[0])
        col = np.random.randint(0, mat.shape[1])
        old_val = mat[row, col]
        # Try nearby half-integer values
        candidates = [old_val + d for d in [-0.5, -0.25, -0.125, 0.125, 0.25, 0.5]]
        for cand in candidates:
            mat[row, col] = cand
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < best_err:
                best_err = err
            else:
                mat[row, col] = old_val
                break
        else:
            continue
        if best_err < tol:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _als_decompose_structured(T, R, n_iter=150, seed=42):
    """
    ALS with structured initialization for the matrix multiplication tensor.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Structured initialization: use the known pattern
    # For rank R, we can try to initialize with patterns that respect the tensor structure
    U = np.zeros((A, R))
    V = np.zeros((B, R))
    W = np.zeros((C, R))
    
    # Initialize with random sparse patterns
    for r in range(R):
        # Each rank component should have some sparsity
        U[:, r] = rng.choice([0, 1, -1], size=A, p=[0.7, 0.15, 0.15])
        V[:, r] = rng.choice([0, 1, -1], size=B, p=[0.7, 0.15, 0.15])
        W[:, r] = rng.choice([0, 1, -1], size=C, p=[0.7, 0.15, 0.15])
    
    for it in range(n_iter):
        # Solve for U
        A_U = np.zeros((B * C, R))
        for r in range(R):
            A_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        
        for a in range(A):
            b_a = T[a, :, :].flatten()
            u, _, _, _ = np.linalg.lstsq(A_U, b_a, rcond=None)
            U[a, :] = u
            
        # Solve for V
        A_V = np.zeros((A * C, R))
        for r in range(R):
            A_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
        for b in range(B):
            b_b = T[:, b, :].flatten()
            v, _, _, _ = np.linalg.lstsq(A_V, b_b, rcond=None)
            V[b, :] = v
            
        # Solve for W
        A_W = np.zeros((A * B, R))
        for r in range(R):
            A_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
        for c in range(C):
            b_c = T[:, :, c].flatten()
            w, _, _, _ = np.linalg.lstsq(A_W, b_c, rcond=None)
            W[c, :] = w
            
        # Check convergence periodically
        if it % 20 == 0 or it == n_iter - 1:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-6:
                break
                
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-4:
        return U, V, W
    return None

def _try_rank(rank, n, m, p, seed=42):
    """Try to find a rank-rank decomposition."""
    T = _build_target_tensor(n, m, p)
    res = _als_decompose_structured(T, rank, n_iter=100, seed=seed)
    
    if res is not None:
        U, V, W = res
        snapped = _snap_to_half_integers(U, V, W)
        if snapped is not None:
            return snapped
            
        # Also try direct rounding
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
            
    return None

def _try_sparse_construction(rank, n, m, p):
    """
    Try to construct a sparse decomposition by enumerating patterns.
    For each rank component r, we assign signs to entries in U, V, W.
    The constraint is that for each (i,j,k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    """
    T = _build_target_tensor(n, m, p)
    A, B, C = T.shape
    
    # We'll try to find a sign pattern that works
    # Start with the schoolbook and try to merge components
    # This is a heuristic: try to find linear dependencies
    
    # Alternative: use integer programming via scipy
    # Set up as: find U,V,W with entries in {-1,0,1} such that tensor matches
    
    # For now, try a randomized search over sparse patterns
    rng = np.random.RandomState(12345)
    for attempt in range(500):
        U = np.zeros((A, rank))
        V = np.zeros((B, rank))
        W = np.zeros((C, rank))
        
        for r in range(rank):
            # Sparse random initialization
            U[:, r] = rng.choice([0, 1, -1], size=A, p=[0.8, 0.1, 0.1])
            V[:, r] = rng.choice([0, 1, -1], size=B, p=[0.8, 0.1, 0.1])
            W[:, r] = rng.choice([0, 1, -1], size=C, p=[0.8, 0.1, 0.1])
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 20 to 24
    # Theoretical lower bound is around 20-22 for this problem
    for rank in [20, 21, 22, 23, 24]:
        if time.time() > _DEADLINE - 5:
            break
            
        # Try sparse construction first (fast)
        if rank <= 22:
            res = _try_sparse_construction(rank, n, m, p)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # Try ALS with multiple seeds
        num_seeds = 30 if rank <= 22 else 10
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 3:
                break
            res = _try_rank(rank, n, m, p, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback: Schoolbook
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
