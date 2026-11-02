"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if valid else 0."""
    return np.ones((n, m, p))

def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct tensor from factor matrices. T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T

def _validate(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _snap_to_half_integers(U, V, W, tol=0.01):
    """Snap values close to half-integers. Returns None if snap is not close enough."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    if (np.max(np.abs(U - U_s)) < tol and
        np.max(np.abs(V - V_s)) < tol and
        np.max(np.abs(W - W_s)) < tol):
        U_s = np.round(U_s * 2) / 2.0
        V_s = np.round(V_s * 2) / 2.0
        W_s = np.round(W_s * 2) / 2.0
        return U_s, V_s, W_s
    return None

def _structured_init(n, m, p, rank, rng):
    """Create structured initialization based on matrix multiplication paths."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each rank-1 term, assign it to a "path" (i, j, k)
    # Use a systematic assignment to cover all necessary multiplications
    paths = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                paths.append((i, j, k))
    
    # Shuffle and assign to rank positions
    rng.shuffle(paths)
    
    for c in range(rank):
        if c < len(paths):
            i, j, k = paths[c]
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        else:
            # For extra rank beyond n*m*p, use random small values
            U[:, c] = rng.randn(n * m) * 0.1
            V[:, c] = rng.randn(m * p) * 0.1
            W[:, c] = rng.randn(p * n) * 0.1
    
    return U, V, W

def _refine_decomposition(U, V, W, n, m, p, rank, max_steps=500, lr=0.05, time_limit=None, start_time=None):
    """Refine a decomposition using gradient descent."""
    if time_limit is None:
        time_limit = 10
    if start_time is None:
        start_time = time.time()
    
    T_target = _build_tensor(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit - 2:
            break
        
        T = _reconstruct(U, V, W, n, m, p, rank)
        err = np.sum((T - T_target)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
        
        # Compute gradients
        dT = 2 * (T - T_target)
        dT_r = dT.reshape(n, m, p)
        
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        U_r = U.reshape(n, m, rank)
        
        dU_r = np.einsum('ijk,jkc,kic->ijc', dT_r, V_r, W_r)
        dV_r = np.einsum('ijk,ijc,kic->jkc', dT_r, U_r, W_r)
        dW_r = np.einsum('ijk,ijc,jkc->kic', dT_r, U_r, V_r)
        
        dU = dU_r.reshape(n*m, rank)
        dV = dV_r.reshape(m*p, rank)
        dW = dW_r.reshape(p*n, rank)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr *= 0.9
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, rank, seed, time_limit):
    """Try to find a valid decomposition of given rank."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Try multiple strategies
    strategies = [
        ('structured', lambda: _structured_init(n, m, p, rank, rng)),
        ('random', lambda: (rng.randn(n*m, rank)*0.3, rng.randn(m*p, rank)*0.3, rng.randn(p*n, rank)*0.3)),
    ]
    
    for strategy_name, init_func in strategies:
        if time.time() - start_time > time_limit - 3:
            break
        
        U, V, W = init_func()
        U, V, W, err = _refine_decomposition(U, V, W, n, m, p, rank, 
                                              max_steps=300, lr=0.03,
                                              time_limit=time_limit - (time.time() - start_time),
                                              start_time=start_time)
        
        if err < 1e-10:
            snapped = _snap_to_half_integers(U, V, W)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                if _validate(U_s, V_s, W_s, n, m, p, rank):
                    return U_s, V_s, W_s
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 145  # Leave margin before 150s kill
    
    # Lower bound on rank
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        rank_time_limit = min(15, time_limit - (time.time() - start_time))
        if rank_time_limit <= 5:
            break
        
        # Try to find decomposition at this rank
        result = _try_rank(n, m, p, rank, seed=rank * 1000 + 42, time_limit=rank_time_limit)
        if result is not None:
            U_s, V_s, W_s = result
            U_list = [[val for val in row] for row in U_s.tolist()]
            V_list = [[val for val in row] for row in V_s.tolist()]
            W_list = [[val for val in row] for row in W_s.tolist()]
            
            # Final validation
            if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
