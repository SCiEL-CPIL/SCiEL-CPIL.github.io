import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

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

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _als_decompose(T, R, n_iter=1000, seed=42, init_type='random', tol=1e-7):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    if init_type == 'random':
        U = rng.randn(A, R) * 0.1
        V = rng.randn(B, R) * 0.1
        W = rng.randn(C, R) * 0.1
    elif init_type == 'sparse_int':
        # Initialize with -1, 0, 1
        U = rng.randint(-1, 2, size=(A, R)).astype(float)
        V = rng.randint(-1, 2, size=(B, R)).astype(float)
        W = rng.randint(-1, 2, size=(C, R)).astype(float)
        # Normalize columns to prevent overflow
        for r in range(R):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-9 and nv > 1e-9 and nw > 1e-9:
                U[:, r] /= nu
                V[:, r] /= nv
                W[:, r] /= nw
    elif init_type == 'schoolbook_trunc':
        n, m, p = 2, 3, 4
        U = np.zeros((A, R))
        V = np.zeros((B, R))
        W = np.zeros((C, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        # Add small random perturbation to remaining columns
        for r in range(col, R):
            U[:, r] = rng.randn(A) * 0.5
            V[:, r] = rng.randn(B) * 0.5
            W[:, r] = rng.randn(C) * 0.5
            
    elif init_type == 'strassen_mix':
        # Attempt to use Strassen-like structure for 2x2x2 parts
        # This is a heuristic initialization
        U = rng.randn(A, R) * 0.5
        V = rng.randn(B, R) * 0.5
        W = rng.randn(C, R) * 0.5
        # Force some sparsity
        mask = rng.rand(A, R) < 0.3
        U[mask] = 0
        mask = rng.rand(B, R) < 0.3
        V[mask] = 0
        mask = rng.rand(C, R) < 0.3
        W[mask] = 0
        
    # Precompute target slices
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        # Solve for U
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
                break
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-6:
        return U, V, W
    return None

def _try_round_and_verify(U, V, W, n, m, p):
    """Try to round to half-integers or integers and verify."""
    # Try half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    if _verify(U_h, V_h, W_h, n, m, p):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    # Try integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try floor/ceil combinations for near-integer values
    for scale in [1, 2]:
        U_s = np.round(U * scale) / scale
        V_s = np.round(V * scale) / scale
        W_s = np.round(W * scale) / scale
        if _verify(U_s, V_s, W_s, n, m, p):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
    # Try a "polish" step if we are very close
    # This is a heuristic: if error is small, try to adjust slightly
    return None

def _polish_solution(U, V, W, T, n, m, p):
    """Use scipy to polish a near-solution to exact values."""
    # Flatten U, V, W
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    
    def objective(x):
        U_x = x[:U.size].reshape(U.shape)
        V_x = x[U.size:U.size+V.size].reshape(V.shape)
        W_x = x[U.size+V.size:].reshape(W.shape)
        T_recon = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
        return np.max(np.abs(T_recon - T))
    
    # This is a max-norm problem, which is non-smooth.
    # We can use least squares on the squared error as a proxy
    def lsq_objective(x):
        U_x = x[:U.size].reshape(U.shape)
        V_x = x[U.size:U.size+V.size].reshape(V.shape)
        W_x = x[U.size+V.size:].reshape(W.shape)
        T_recon = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
        return np.sum((T_recon - T)**2)
        
    try:
        res = scipy.optimize.least_squares(lsq_objective, x0=np.concatenate([U_flat, V_flat, W_flat]), method='trf', max_nfev=100)
        if res.success:
            U_p = res.x[:U.size].reshape(U.shape)
            V_p = res.x[U.size:U.size+V.size].reshape(V.shape)
            W_p = res.x[U.size+V.size:].reshape(W.shape)
            
            # Try rounding the polished solution
            result = _try_round_and_verify(U_p, V_p, W_p, n, m, p)
            if result:
                return result
    except:
        pass
    return None

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 22 up to 23
    # 24 is the fallback
    for rank in [23, 22]:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple seeds and initializations
        num_seeds = 30 if rank == 23 else 20
        init_types = ['random', 'sparse_int', 'schoolbook_trunc', 'strassen_mix']
        
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 5:
                break
            
            for init_type in init_types:
                if time.time() > _DEADLINE - 3:
                    break
                
                res = _als_decompose(T, rank, n_iter=300, seed=seed, init_type=init_type, tol=1e-8)
                if res is not None:
                    U, V, W = res
                    result = _try_round_and_verify(U, V, W, n, m, p)
                    if result is not None:
                        return {"U": result[0], "V": result[1], "W": result[2]}
                    
                    # If ALS converged but rounding failed, try polishing
                    result = _polish_solution(U, V, W, T, n, m, p)
                    if result is not None:
                        return {"U": result[0], "V": result[1], "W": result[2]}
        
    # Fallback: Schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
