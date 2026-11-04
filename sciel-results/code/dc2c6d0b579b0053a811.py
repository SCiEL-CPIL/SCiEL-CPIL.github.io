import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _fast_verify(U, V, W, n=3, m=3, p=3):
    """Fast verification using tensor contraction."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    r = U_arr.shape[1]
    
    U_reshaped = U_arr.reshape(n, m, r)
    V_reshaped = V_arr.reshape(m, p, r)
    W_reshaped = W_arr.reshape(p, n, r)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(result, 1.0, atol=1e-9)


def _construct_schoolbook(n=3, m=3, p=3):
    """Construct the schoolbook rank-(n*m*p) decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _compute_error(U, V, W, n=3, m=3, p=3):
    """Compute the sum of squared errors."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.sum((result - 1.0) ** 2)


def _try_rank_r(r, n=3, m=3, p=3, time_budget=50.0, seed=42):
    """Try to find a rank-r decomposition using local search with half-integer entries."""
    start = time.time()
    np.random.seed(seed)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook pattern for first min(r, 27) columns
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    
    nmp = n * m * p
    for c in range(min(r, nmp)):
        i = (c // (m * p)) % n
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # Random initialization for remaining columns
    for c in range(min(r, nmp), r):
        U[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n * m)
        V[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=m * p)
        W[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=p * n)
    
    for iteration in range(150000):
        if time.time() - start > time_budget:
            break
        
        error = _compute_error(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_error < 1e-9:
                # Try to round to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if _fast_verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
                return best_U, best_V, best_W
        
        # Random move
        choice = np.random.randint(3)
        if choice == 0:
            idx = np.random.randint(n * m)
            col = np.random.randint(r)
            old_val = U[idx, col]
            candidates = [v for v in [-1.0, -0.5, 0.0, 0.5, 1.0] if abs(v - old_val) > 1e-10]
            if candidates:
                U[idx, col] = np.random.choice(candidates)
        elif choice == 1:
            idx = np.random.randint(m * p)
            col = np.random.randint(r)
            old_val = V[idx, col]
            candidates = [v for v in [-1.0, -0.5, 0.0, 0.5, 1.0] if abs(v - old_val) > 1e-10]
            if candidates:
                V[idx, col] = np.random.choice(candidates)
        else:
            idx = np.random.randint(p * n)
            col = np.random.randint(r)
            old_val = W[idx, col]
            candidates = [v for v in [-1.0, -0.5, 0.0, 0.5, 1.0] if abs(v - old_val) > 1e-10]
            if candidates:
                W[idx, col] = np.random.choice(candidates)
    
    if best_error < 1e-9 and best_U is not None:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _fast_verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        return best_U, best_V, best_W
    
    return None


def construct():
    start_time = time.time()
    timeout = 130.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-27
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try rank 23, 24, 25, 26
    for target_rank in [23, 24, 25, 26]:
        if time.time() - start_time < timeout - 10:
            time_budget = min(30.0, timeout - (time.time() - start_time) - 10)
            result = _try_rank_r(target_rank, n, m, p, time_budget, seed=target_rank * 100)
            if result is not None:
                U, V, W = result
                if _fast_verify(U, V, W, n, m, p):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
