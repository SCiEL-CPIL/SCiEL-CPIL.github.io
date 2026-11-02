import time
import numpy as np

_DEADLINE = 150.0

def _verify_numpy(U, V, W, n, m, p, atol=1e-8):
    """Verify decomposition using numpy tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # T[u_idx, v_idx, w_idx] = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    # u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i
    # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 otherwise
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, target, atol=atol)

def _build_rank_44():
    """Build the known rank-44 decomposition for 2x3 x 3x8."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8, rank 16
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8
    # Split into 4 blocks of 2x2 x 2x2, each using Strassen rank 7
    for block in range(4):
        k_off = block * 2
        # M1 = (a00+a11)(b00+b11)
        U[0 * m + 1][col] = 1   # i=0, j=1
        U[1 * m + 2][col] = 1   # i=1, j=2
        V[1 * p + k_off][col] = 1       # j=1, k=k_off
        V[2 * p + k_off + 1][col] = 1   # j=2, k=k_off+1
        W[k_off * n + 0][col] = 1
        W[(k_off + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[1 * m + 1][col] = 1   # i=1, j=1
        U[1 * m + 2][col] = 1   # i=1, j=2
        V[1 * p + k_off][col] = 1
        W[k_off * n + 1][col] = 1
        W[(k_off + 1) * n + 1][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[0 * m + 1][col] = 1
        V[1 * p + k_off + 1][col] = 1
        V[2 * p + k_off + 1][col] = -1
        W[(k_off + 1) * n + 0][col] = 1
        W[(k_off + 1) * n + 1][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[1 * m + 2][col] = 1
        V[2 * p + k_off][col] = 1
        V[1 * p + k_off][col] = -1
        W[k_off * n + 0][col] = 1
        W[k_off * n + 1][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k_off + 1][col] = 1
        W[k_off * n + 0][col] = -1
        W[(k_off + 1) * n + 0][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k_off][col] = 1
        V[1 * p + k_off + 1][col] = 1
        W[(k_off + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k_off][col] = 1
        V[2 * p + k_off + 1][col] = 1
        W[k_off * n + 0][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W, rank

def _attempt_rank_reduction(U44, V44, W44, target_rank, n, m, p, deadline):
    """Attempt to reduce rank from 44 to target_rank via numerical optimization."""
    if time.time() > deadline - 10:
        return None
    
    U44 = np.array(U44, dtype=float)
    V44 = np.array(V44, dtype=float)
    W44 = np.array(W44, dtype=float)
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def residuals(params):
        r = target_rank
        U = params[:r * n*m].reshape(r, n*m).T
        V = params[r * n*m : r * (n*m + m*p)].reshape(r, m*p).T
        W = params[r * (n*m + m*p):].reshape(r, p*n).T
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        return (T - target).flatten()
    
    # Try multiple strategies to initialize
    strategies = []
    
    # Strategy 1: Drop last term from rank 44
    U_init = U44[:, :target_rank].copy()
    V_init = V44[:, :target_rank].copy()
    W_init = W44[:, :target_rank].copy()
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    strategies.append(('drop_last', x0))
    
    # Strategy 2: Random perturbation of the dropped term
    if target_rank < 44:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            U_init = U44[:, :target_rank].copy()
            V_init = V44[:, :target_rank].copy()
            W_init = W44[:, :target_rank].copy()
            # Add small random perturbation
            noise_scale = 0.1
            U_init += np.random.randn(n*m, target_rank) * noise_scale
            V_init += np.random.randn(m*p, target_rank) * noise_scale
            W_init += np.random.randn(p*n, target_rank) * noise_scale
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            strategies.append((f'perturb_{attempt}', x0))
    
    best_result = None
    best_residual = float('inf')
    
    for name, x0 in strategies:
        if time.time() > deadline - 5:
            break
        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, x0, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.fun is not None and np.linalg.norm(result.fun) < best_residual:
                best_residual = np.linalg.norm(result.fun)
                best_result = result
                if best_residual < 1e-10:
                    break
        except Exception:
            continue
    
    if best_result is not None and best_residual < 1e-10:
        r = target_rank
        U = best_result.x[:r * n*m].reshape(r, n*m).T
        V = best_result.x[r * n*m : r * (n*m + m*p)].reshape(r, m*p).T
        W = best_result.x[r * (n*m + m*p):].reshape(r, p*n).T
        
        # Verify with numpy
        if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # 5s safety margin
    
    # Build the rank-44 decomposition
    U44, V44, W44, rank44 = _build_rank_44()
    
    # Verify it works
    if not _verify_numpy(U44, V44, W44, n, m, p):
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce to rank 43
    if time.time() < deadline - 15:
        result_43 = _attempt_rank_reduction(U44, V44, W44, 43, n, m, p, deadline)
        if result_43 is not None:
            return result_43
    
    # Try to reduce to rank 42 if time permits
    if time.time() < deadline - 10:
        result_42 = _attempt_rank_reduction(U44, V44, W44, 42, n, m, p, deadline)
        if result_42 is not None:
            return result_42
    
    # Return rank 44
    return {"U": U44, "V": V44, "W": W44}
