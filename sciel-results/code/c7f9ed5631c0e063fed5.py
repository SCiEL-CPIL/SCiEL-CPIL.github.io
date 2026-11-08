"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _objective(x, n, m, p, rank):
    """Compute the sum of squared errors for a given parameter vector."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    err_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                err_sum += (val - 1.0) ** 2
    return err_sum

def _structured_init_block(n, m, p, rank, seed):
    """
    Initialize with block structure for 3x3x6.
    Split rank into two halves for the two 3x3x3 sub-problems.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r1 = rank // 2
    r2 = rank - r1
    
    # Block 1: cols 0-2 of B
    # U: all rows
    # V: rows where k in 0..2 -> indices j*6 + k
    # W: rows where k in 0..2 -> indices k*3 + i
    
    U[:, :r1] = np.random.randn(n * m, r1) * 0.5
    V_indices_1 = [j * p + k for j in range(m) for k in range(3)]
    W_indices_1 = [k * n + i for k in range(3) for i in range(n)]
    
    V[V_indices_1, :r1] = np.random.randn(len(V_indices_1), r1) * 0.5
    W[W_indices_1, :r1] = np.random.randn(len(W_indices_1), r1) * 0.5
    
    # Block 2: cols 3-5 of B
    # U: all rows
    # V: rows where k in 3..5 -> indices j*6 + k
    # W: rows where k in 3..5 -> indices k*3 + i
    
    U[:, r1:] = np.random.randn(n * m, r2) * 0.5
    V_indices_2 = [j * p + k for j in range(m) for k in range(3, 6)]
    W_indices_2 = [k * n + i for k in range(3, 6) for i in range(n)]
    
    V[V_indices_2, r1:] = np.random.randn(len(V_indices_2), r2) * 0.5
    W[W_indices_2, r1:] = np.random.randn(len(W_indices_2), r2) * 0.5
    
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, max_time=10.0):
    """
    Try to find a rank-target_rank decomposition using local optimization
    with structured initialization.
    """
    start_time = time.time()
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > max_time:
            break
            
        U_init, V_init, W_init = _structured_init_block(n, m, p, target_rank, seed)
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # L-BFGS-B is efficient for this smooth objective
            result = minimize(_objective, x0, args=(n, m, p, target_rank),
                              method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < 1e-10:
                U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                
                err = _check_decomposition(U_found, V_found, W_found, n, m, p)
                if err < 1e-8:
                    return U_found, V_found, W_found
        except Exception:
            continue
            
    return None, None, None

def _high_precision_polish(U, V, W, n, m, p, max_iter=500):
    """
    High-precision polish step.
    """
    rank = U.shape[1]
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        return _objective(x, n, m, p, rank)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < 1e-12:
            U_polished = result.x[:n*m*rank].reshape(n * m, rank)
            V_polished = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
            W_polished = result.x[n*m*rank + m*p*rank :].reshape(p * n, rank)
            
            err = _check_decomposition(U_polished, V_polished, W_polished, n, m, p)
            if err < 1e-10:
                return U_polished, V_polished, W_polished
    except Exception:
        pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is valid
    err = _check_decomposition(U, V, W, n, m, p)
    if err >= 1e-10:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try to find lower rank decompositions
    # Lower bound for 3x3x6 is at least 18 (trivial) but likely higher.
    # Schoolbook is 54.
    # Try ranks from 50 down to 30.
    # We prioritize ranks that are multiples of 2 or 3 due to structure, but try all.
    
    for target_rank in range(50, 29, -1):
        if time.time() - start_time > deadline - 5:
            break
        
        # Allocate time based on rank: higher ranks are easier, give less time?
        # Actually, lower ranks are harder, but we stop at first success.
        # Give 15s per rank attempt.
        max_time_per_rank = 15.0
        
        U_r, V_r, W_r = _try_find_decomposition(n, m, p, target_rank, max_time=max_time_per_rank)
        
        if U_r is not None:
            # High-precision polish
            U_r, V_r, W_r = _high_precision_polish(U_r, V_r, W_r, n, m, p)
            
            # Try to round to half-integers
            U_rounded = _to_half_int(U_r)
            V_rounded = _to_half_int(V_r)
            W_rounded = _to_half_int(W_r)
            
            if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            else:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
            
            break
    
    # Final result
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
