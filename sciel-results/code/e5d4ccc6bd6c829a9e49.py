import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) > tol:
        return False
    return True

def _check_half_integer(X, tol=1e-6):
    """Check if all entries are close to half-integers."""
    X2 = X * 2
    return np.allclose(X2, np.round(X2), atol=tol)

def _round_to_half_int(X):
    """Round to nearest half-integer."""
    return np.round(X * 2) / 2

def _verify_and_round(U, V, W, T, tol=1e-8):
    """Try to round to half-integers and verify the decomposition."""
    U_h = _round_to_half_int(U)
    V_h = _round_to_half_int(V)
    W_h = _round_to_half_int(W)
    
    if _verify_decomposition(U_h, V_h, W_h, T, tol=tol):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    # Try integers as well
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify_decomposition(U_i, V_i, W_i, T, tol=tol):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try local perturbation around rounded values
    U_r = np.round(U * 2)
    V_r = np.round(V * 2)
    W_r = np.round(W * 2)
    
    T_recon = np.einsum('ar,br,cr->abc', U_r/2, V_r/2, W_r/2)
    err = np.max(np.abs(T_recon - T))
    
    if err < 2.0:
        rng = np.random.RandomState(12345)
        for _ in range(50):
            if time.time() > _DEADLINE - 3:
                break
            U_t = U_r.copy()
            V_t = V_r.copy()
            W_t = W_r.copy()
            
            # Perturb a few random entries by ±1 (in half-integer units)
            for _ in range(4):
                mat_list = [U_t, V_t, W_t]
                mat_idx = rng.randint(3)
                mat = mat_list[mat_idx]
                idx = rng.randint(mat.size)
                r, c = np.unravel_index(idx, mat.shape)
                mat[r, c] += rng.choice([-1, 1])
            
            if _verify_decomposition(U_t/2, V_t/2, W_t/2, T, tol=1e-6):
                return (U_t/2).tolist(), (V_t/2).tolist(), (W_t/2).tolist()
    
    return None

def _als_decompose(T, R, n_iter=500, seed=42, tol=1e-9, time_limit=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(A, R) * 0.5
    V = rng.randn(B, R) * 0.5
    W = rng.randn(C, R) * 0.5
    
    # Precompute target slices
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    start_time = time.time()
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        if time_limit and time.time() - start_time > time_limit:
            break
        
        # Solve for U: for each a, T_a[a] = U[a,:] @ (V*W)
        A_U = np.outer(V.flatten(), W.flatten()).reshape(B * C, R)
        for a in range(A):
            U[a, :], _, _, _ = np.linalg.lstsq(A_U, T_a[a], rcond=None)
        
        # Solve for V
        A_V = np.outer(U.flatten(), W.flatten()).reshape(A * C, R)
        for b in range(B):
            V[b, :], _, _, _ = np.linalg.lstsq(A_V, T_b[b], rcond=None)
        
        # Solve for W
        A_W = np.outer(U.flatten(), V.flatten()).reshape(A * B, R)
        for c in range(C):
            W[c, :], _, _, _ = np.linalg.lstsq(A_W, T_c[c], rcond=None)
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-7:
        return U, V, W
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 12 up to 23
    # Lower is better, so try 12 first
    # Allocate time budget: more time for lower ranks
    ranks_to_try = list(range(12, 24))
    
    # Time allocation: total ~135s available
    # Give more time to lower ranks
    time_budgets = {
        12: 40, 13: 25, 14: 20, 15: 15,
        16: 10, 17: 8, 18: 6, 19: 5,
        20: 4, 21: 3, 22: 2, 23: 2
    }
    
    for rank in ranks_to_try:
        if time.time() > _DEADLINE - 10:
            break
        
        budget = time_budgets.get(rank, 2)
        time_per_attempt = 0.3  # ~300ms per ALS run
        num_attempts = max(1, int(budget / time_per_attempt))
        
        for seed in range(num_attempts):
            if time.time() > _DEADLINE - 5:
                break
            
            res = _als_decompose(T, rank, n_iter=300, seed=seed, tol=1e-9)
            if res is not None:
                U, V, W = res
                result = _verify_and_round(U, V, W, T)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
        
        # Also try with different initialization strategies for this rank
        for init_seed in range(5):
            if time.time() > _DEADLINE - 5:
                break
            # Use a different random seed offset
            alt_seed = seed * 1000 + init_seed
            res = _als_decompose(T, rank, n_iter=200, seed=alt_seed, tol=1e-9)
            if res is not None:
                U, V, W = res
                result = _verify_and_round(U, V, W, T)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
