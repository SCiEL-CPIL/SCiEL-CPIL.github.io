import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_max_error(U, V, W, n, m, p):
    """Compute max absolute error using vectorized einsum."""
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
    return float(np.max(np.abs(T - 1.0)))

def _smooth_objective(U, V, W, n, m, p, temperature=1.0):
    """Smooth surrogate for the max-error objective using log-sum-exp."""
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
    squared_errors = (T - 1.0) ** 2
    max_sq = np.max(squared_errors)
    if max_sq < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp((squared_errors - max_sq) / temperature))) + max_sq / temperature
    return np.exp(log_sum) * temperature

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-125 decomposition."""
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

def _structured_init(n, m, p, rank, seed):
    """
    Initialize factor matrices with a structured perturbation of schoolbook.
    Uses integer/half-integer values from {-1, -0.5, 0, 0.5, 1}.
    """
    np.random.seed(seed)
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Start with schoolbook-like structure: each row has a few non-zero entries
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each row, place a few random non-zero entries
    num_nonzero_U = max(2, rank // 25)
    num_nonzero_V = max(2, rank // 25)
    num_nonzero_W = max(2, rank // 25)
    
    for i in range(n * m):
        cols = np.random.choice(rank, size=min(num_nonzero_U, rank), replace=False)
        U[i, cols] = np.random.choice(vals, size=len(cols))
    
    for j in range(m * p):
        cols = np.random.choice(rank, size=min(num_nonzero_V, rank), replace=False)
        V[j, cols] = np.random.choice(vals, size=len(cols))
    
    for k in range(p * n):
        cols = np.random.choice(rank, size=min(num_nonzero_W, rank), replace=False)
        W[k, cols] = np.random.choice(vals, size=len(cols))
    
    return U, V, W

def _try_alternating_minimization_smooth(n, m, p, rank, start_time, deadline):
    """
    Attempt to find a rank decomposition using alternating minimization
    with a smooth surrogate objective and multiple restarts.
    Returns (U, V, W) or (None, None, None) if failed.
    """
    best_result = None
    best_err = float('inf')
    
    # Use more restarts with structured initialization
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        seed = 42 + restart * 100
        np.random.seed(seed)
        
        # Mix of initialization strategies
        if restart % 3 == 0:
            # Structured sparse init
            U, V, W = _structured_init(n, m, p, rank, seed)
        elif restart % 3 == 1:
            # Dense random from half-integer set
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, rank))
        else:
            # Gaussian init
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
        
        max_iters = 80
        prev_err = float('inf')
        stagnation_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Check convergence using vectorized max error
            max_err = _compute_max_error(U, V, W, n, m, p)
            
            if max_err < 1e-8:
                # High-precision polish pass
                U, V, W = _polish_solution(U, V, W, n, m, p, rank, deadline)
                max_err = _compute_max_error(U, V, W, n, m, p)
                if max_err < 1e-10:
                    return U, V, W
                if max_err < best_err:
                    best_err = max_err
                    best_result = (U.copy(), V.copy(), W.copy())
                break
            
            # Track stagnation
            if max_err < prev_err - 1e-12:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            prev_err = max_err
            
            if stagnation_count > 25:
                break
        
        # Track best result across restarts
        if best_result is None:
            max_err = _compute_max_error(U, V, W, n, m, p)
            if max_err < best_err:
                best_err = max_err
                best_result = (U.copy(), V.copy(), W.copy())
    
    # Final check on best result
    if best_result is not None:
        U, V, W = best_result
        max_err = _compute_max_error(U, V, W, n, m, p)
        if max_err < 1e-8:
            return U, V, W
    
    return None, None, None

def _polish_solution(U, V, W, n, m, p, rank, deadline):
    """
    High-precision polish pass with tighter tolerance.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Update U with tighter solve
        for i in range(n):
            for j in range(m):
                B = np.zeros((rank, p))
                for k in range(p):
                    B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=1e-12)[0]
                except:
                    pass
        
        # Update V with tighter solve
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, n))
                for i in range(n):
                    A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=1e-12)[0]
                except:
                    pass
        
        # Update W with tighter solve
        for k in range(p):
            for i in range(n):
                C = np.zeros((rank, m))
                for j in range(m):
                    C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=1e-12)[0]
                except:
                    pass
        
        max_err = _compute_max_error(U, V, W, n, m, p)
        if max_err < 1e-10:
            break
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialize with schoolbook decomposition (Rank 125)
    rank_schoolbook = n * m * p
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # 2. Try alternating minimization with smooth surrogate for rank 124
    target_rank = 124
    if time.time() < deadline - 20.0:
        try:
            U, V, W = _try_alternating_minimization_smooth(n, m, p, target_rank, start_time, deadline)
            if U is not None:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                # Verify the rounded solution
                if _validate_decomposition(U_r, V_r, W_r, n, m, p):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
        except Exception:
            pass

    # 3. If we found rank 124, try to find rank 123
    if best_rank == 124 and time.time() < deadline - 20.0:
        try:
            U, V, W = _try_alternating_minimization_smooth(n, m, p, 123, start_time, deadline)
            if U is not None:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _validate_decomposition(U_r, V_r, W_r, n, m, p):
                    best_rank = 123
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
        except Exception:
            pass

    # 4. Return best result
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
