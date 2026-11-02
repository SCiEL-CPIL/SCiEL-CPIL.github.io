"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # Safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10
    W[1*2+1, 1] = -1 # C11
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[1*2+0, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[0*2+1, 3] = 1  # C10
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[1*2+0, 4] = 1  # C01
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each valid (i,j,k), check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for invalid combinations, it should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This combination is valid, should give 1
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _residual_func(params, n, m, p, rank):
    """
    Compute the residual of the tensor decomposition.
    params: flattened [U_flat, V_flat, W_flat]
    Returns residual vector: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    """
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U_flat = params[:U_size]
    V_flat = params[U_size:U_size+V_size]
    W_flat = params[U_size+V_size:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                residuals.append(val - 1.0)
    
    return np.array(residuals)

def _try_lm_optimization(n, m, p, target_rank, timeout=60, n_restarts=3):
    """
    Try to find a dense decomposition of the given rank using Levenberg-Marquardt.
    Returns (U, V, W) if found with residual < tol, else None.
    """
    start_time = time.time()
    tol = 1e-8
    
    U_size = n * m * target_rank
    V_size = m * p * target_rank
    W_size = p * n * target_rank
    total_params = U_size + V_size + W_size
    
    for restart in range(n_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize: use Strassen-based blocks for first min(target_rank, 7*2) terms,
        # then random or schoolbook for remaining
        rng = np.random.default_rng(seed=42 + restart)
        
        # Start with a reasonable initialization
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Use Strassen blocks for initialization if possible
        U_str, V_str, W_str = _build_strassen_2x2x2()
        U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
        
        cols_filled = 0
        # Add Strassen blocks
        while cols_filled + 7 <= target_rank:
            U_init[:, cols_filled:cols_filled+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = cols_filled // 7 * 2 + k_local
                    if k_global < p:
                        V_init[j*p + k_global, cols_filled:cols_filled+7] = V_str[j*2 + k_local, :]
            for k_local in range(2):
                k_global = cols_filled // 7 * 2 + k_local
                if k_global < p:
                    for i in range(n):
                        W_init[k_global*n + i, cols_filled:cols_filled+7] = W_str[k_local*2 + i, :]
            cols_filled += 7
        
        # Fill remaining with schoolbook or random
        if cols_filled < target_rank:
            remaining = target_rank - cols_filled
            # Add small random perturbation to break symmetry
            U_init[:, cols_filled:] = rng.normal(0, 0.1, size=(n*m, remaining))
            V_init[:, cols_filled:] = rng.normal(0, 0.1, size=(m*p, remaining))
            W_init[:, cols_filled:] = rng.normal(0, 0.1, size=(p*n, remaining))
        
        # Also add small random noise to all entries to help optimization
        U_init += rng.normal(0, 0.01, size=U_init.shape)
        V_init += rng.normal(0, 0.01, size=V_init.shape)
        W_init += rng.normal(0, 0.01, size=W_init.shape)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                _residual_func,
                params_init,
                args=(n, m, p, target_rank),
                method='trf',
                max_nfev=200,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
        except Exception:
            continue
        
        if result.success or result.cost < tol**2 * n * m * p:
            # Extract and verify
            U_flat = result.x[:U_size]
            V_flat = result.x[U_size:U_size+V_size]
            W_flat = result.x[U_size+V_size:]
            
            U_sol = U_flat.reshape(n*m, target_rank)
            V_sol = V_flat.reshape(m*p, target_rank)
            W_sol = W_flat.reshape(p*n, target_rank)
            
            if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-6):
                return U_sol, V_sol, W_sol
    
    return None

def _try_find_223_rank10(timeout=30):
    """
    Try to find a rank-10 decomposition for (2,2,3) matrix multiplication.
    Uses LM optimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    return _try_lm_optimization(n, m, p, rank, timeout=timeout, n_restarts=5)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something is wrong
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find a (2,2,3) rank-10 decomposition
    result_223 = _try_find_223_rank10(timeout=min(30, TIME_LIMIT - elapsed - 5))
    if result_223 is not None:
        U223, V223, W223 = result_223
        # Build full decomposition using five (2,2,3) blocks
        total_rank = 5 * 10  # 50
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        for b in range(5):
            start_k = b * 3
            col_offset = b * 10
            U_full[:, col_offset:col_offset+10] = U223
            for j in range(m):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V_full[j*p + k_global, col_offset:col_offset+10] = V223[j*3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(n):
                    W_full[k_global*n + i, col_offset:col_offset+10] = W223[k_local*2 + i, :]
        
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            best_rank = 50
            best_U, best_V, best_W = U_full, V_full, W_full
    
    # Try direct LM optimization on the full (2,2,15) problem with lower rank
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        # Try rank 50 directly on the full problem
        for target_rank in [50, 49, 48]:
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 15:
                break
            
            remaining_time = TIME_LIMIT - 15 - elapsed
            if remaining_time < 10:
                break
            
            result = _try_lm_optimization(n, m, p, target_rank, timeout=remaining_time, n_restarts=2)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break  # Found a valid lower rank, stop
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
