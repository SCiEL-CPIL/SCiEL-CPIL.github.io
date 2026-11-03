import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 135.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    All other entries are 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _is_exact_solution(U, V, W, T, tol=1e-10):
    """Check if the decomposition is exact within tolerance."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    diff = reconstructed - T
    return np.max(np.abs(diff)) < tol

def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2 where U,V,W are 2D arrays (rows x rank).
    The decomposition is T = sum_r U[:,r] V[:,r] W[:,r]^T
    which in matrix form for the first two indices is:
    T[:,:,c] = U @ (V * W[:,c])^T for each c? No.
    
    Actually T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    So for fixed k, the matrix M_k of size (n*m x m*p) is U @ diag(V[:,k]??) no.
    
    Let's reshape: T is (A, B, C).
    The decomposition means T = sum_r U[:,r] outer V[:,r] outer W[:,r].
    We can compute the error by iterating over r, or by using einsum.
    np.einsum('ar,br,cr->abc', U, V, W)
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Create structured initializations that might be closer to a low-rank solution.
    Uses multiple strategies: random, structured, and hybrid.
    """
    rng = np.random.RandomState(seed)
    strategy = seed % 4
    
    if strategy == 0:
        # Pure random with normalization
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
    elif strategy == 1:
        # Structured: initialize based on matrix multiplication pattern
        # Each term in the sum should correspond to a (i,j,k) triple
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Distribute the n*m*p = 50 required terms across rank factors
        terms_per_factor = max(1, (n*m*p + rank - 1) // rank)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    # Add some randomness to the structured pattern
                    scale = 1.0 + 0.1 * rng.randn()
                    U[i*m + j, col] = scale
                    V[j*p + k, col] = scale
                    W[k*n + i, col] = 1.0 / (scale * scale)
                    col += 1
        # Fill remaining columns with small random values
        if col < rank:
            U[:, col:] = 0.01 * rng.randn(n*m, rank - col)
            V[:, col:] = 0.01 * rng.randn(m*p, rank - col)
            W[:, col:] = 0.01 * rng.randn(p*n, rank - col)
    elif strategy == 2:
        # Hybrid: start from schoolbook-like structure but with perturbations
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Try to pack multiple (i,j,k) terms into each factor
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    noise_u = 0.05 * rng.randn()
                    noise_v = 0.05 * rng.randn()
                    noise_w = 0.05 * rng.randn()
                    U[i*m + j, col] = 1.0 + noise_u
                    V[j*p + k, col] = 1.0 + noise_v
                    W[k*n + i, col] = 1.0 + noise_w
                    col += 1
        if col < rank:
            U[:, col:] = 0.01 * rng.randn(n*m, rank - col)
            V[:, col:] = 0.01 * rng.randn(m*p, rank - col)
            W[:, col:] = 0.01 * rng.randn(p*n, rank - col)
    else:
        # Sparse random: most entries zero, some random
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        for r in range(rank):
            # Each factor column has a few non-zero entries
            nz_u = rng.choice(n*m, size=min(3, n*m), replace=False)
            nz_v = rng.choice(m*p, size=min(5, m*p), replace=False)
            nz_w = rng.choice(p*n, size=min(5, p*n), replace=False)
            
            U[nz_u, r] = rng.randn(len(nz_u))
            V[nz_v, r] = rng.randn(len(nz_v))
            W[nz_w, r] = rng.randn(len(nz_w))
    
    # Normalize columns to prevent numerical issues
    for mat in [U, V, W]:
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        mat /= norms
    
    return U, V, W

def _als_step(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration with improved numerical stability.
    """
    A, B, C = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Update U: for each a, solve sum_r U[a,r] * (V[:,r] * W[:,r])[b,c] = T[a,b,c]
    # Flatten: for fixed a, the equation is: sum_r U[a,r] * M_r[b,c] = T[a,b,c]
    # where M_r[b,c] = V[b,r] * W[c,r]
    # Reshape to 2D: M_flat has shape (B*C, rank), T_flat has shape (A, B*C)
    VW = np.einsum('br,cr->bcr', V, W)  # (B, C, rank)
    VW_flat = VW.transpose(2, 0, 1).reshape(rank, B*C)  # (rank, B*C)
    T_flat = T.reshape(A, B*C)  # (A, B*C)
    
    # Solve: VW_flat @ U.T = T_flat  =>  U.T = lstsq(VW_flat.T, T_flat.T)
    # Actually: sum_r U[a,r] * VW_flat[r, bc] = T_flat[a, bc]
    # So: VW_flat.T @ U.T = T_flat.T
    if rank > 0:
        U_new, _, _, _ = lstsq(VW_flat.T, T_flat.T, cond=None)
        U = U_new.T  # (A, rank)
    
    # Update V
    UW = np.einsum('ar,cr->acr', U, W)  # (A, C, rank)
    UW_flat = UW.transpose(2, 0, 1).reshape(rank, A*C)  # (rank, A*C)
    T_flat_V = T.transpose(1, 0, 2).reshape(B, A*C)  # (B, A*C)
    
    if rank > 0:
        V_new, _, _, _ = lstsq(UW_flat.T, T_flat_V.T, cond=None)
        V = V_new.T  # (B, rank)
    
    # Update W
    UV = np.einsum('ar,br->abr', U, V)  # (A, B, rank)
    UV_flat = UV.transpose(2, 0, 1).reshape(rank, A*B)  # (rank, A*B)
    T_flat_W = T.transpose(2, 0, 1).reshape(C, A*B)  # (C, A*B)
    
    if rank > 0:
        W_new, _, _, _ = lstsq(UV_flat.T, T_flat_W.T, cond=None)
        W = W_new.T  # (C, rank)
    
    return U, V, W

def _try_rank(n, m, p, rank, T, seed=0, time_limit=10.0):
    """
    Attempt to find a decomposition of the given rank using multiple strategies.
    Returns (U, V, W, error) if successful, or None.
    """
    start_time = time.time()
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    # Try multiple initializations and optimization paths
    for init_seed in range(4):
        if time.time() - start_time > time_limit:
            break
        
        current_seed = seed * 100 + init_seed
        U, V, W = _structured_init(n, m, p, rank, current_seed)
        
        err = _tensor_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Run ALS iterations
        for iteration in range(500):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _als_step(U, V, W, T, n, m, p)
            
            err = _tensor_error(U, V, W, T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Early termination if converged
            if best_err < 1e-14:
                break
    
    if best_U is None:
        return None, None, None, float('inf')
    
    return best_U, best_V, best_W, best_err

def _gradient_descent_polish(U, V, W, T, n, m, p, rank, time_limit=5.0):
    """
    Polish a near-converged solution using more ALS iterations with tighter convergence.
    """
    start_time = time.time()
    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        U, V, W = _als_step(U, V, W, T, n, m, p)
        err = _tensor_error(U, V, W, T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-14:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    T = _build_tensor(n, m, p)
    
    found_rank = n * m * p
    found_U, found_V, found_W = U, V, W
    
    # Try ranks in a smart order: start from known good ranks and work down
    # For 2x5x5, the Strassen-like rank is likely around 30-40
    # Try: 45, 40, 35, 30, 28, 27, 26, 25
    ranks_to_try = [45, 40, 35, 30, 28, 27, 26, 25]
    
    # Add remaining ranks if time permits
    for rank in range(49, 24, -1):
        if rank not in ranks_to_try:
            ranks_to_try.append(rank)
    
    # Sort: try promising ranks first, then others
    ranks_to_try = sorted(set(ranks_to_try), key=lambda r: (abs(r - 35), r))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        if rank >= found_rank:
            continue
        
        # Try multiple seeds for this rank
        success = False
        max_seeds = 5 if rank <= 35 else 3
        for seed in range(max_seeds):
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            
            time_limit = min(15.0, max(5.0, (DEADLINE_SECONDS - 5 - (time.time() - start_time)) / (max_seeds - seed)))
            try:
                U_new, V_new, W_new, err = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if U_new is None:
                    continue
                
                if err < 1e-10:
                    # Check for exactness with tighter tolerance
                    if _is_exact_solution(U_new, V_new, W_new, T, tol=1e-10):
                        # Polish with more iterations
                        U_pol, V_pol, W_pol, err_pol = _gradient_descent_polish(
                            U_new, V_new, W_new, T, n, m, p, rank, time_limit=5.0
                        )
                        if err_pol < 1e-10 and _is_exact_solution(U_pol, V_pol, W_pol, T, tol=1e-10):
                            found_rank = rank
                            found_U, found_V, found_W = U_pol, V_pol, W_pol
                            success = True
                            break
            except Exception:
                continue
        
        if success:
            continue
    
    # Return the best found
    U_list = found_U.tolist()
    V_list = found_V.tolist()
    W_list = found_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
