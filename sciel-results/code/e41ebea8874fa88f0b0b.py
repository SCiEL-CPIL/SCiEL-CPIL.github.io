"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s limit

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _compute_tensor_fast(U, V, W, n, m, p):
    """Compute the decomposed tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] efficiently."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Use einsum for efficient contraction
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _error(T):
    """Compute max absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))

def _als_iteration(U, V, W, n, m, p):
    """
    One ALS iteration: solve for each factor given the other two.
    Returns new U, V, W.
    """
    rank = U.shape[1]
    U_new = np.zeros((n * m, rank))
    V_new = np.zeros((m * p, rank))
    W_new = np.zeros((p * n, rank))
    
    # Solve for U: for each (i,j), find U[i*m+j,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b_vec = np.ones(p)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Solve for V: for each (j,k), find V[j*p+k,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b_vec = np.ones(n)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Solve for W: for each (k,i), find W[k*n+i,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b_vec = np.ones(m)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _als_with_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Run ALS iterations with polish."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor_fast(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor_fast(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Polish pass: tighter tolerance, more iterations
    if best_err < 0.1 and time.time() - _START_TIME < _TIMEOUT - 10:
        U, V, W = best_U, best_V, best_W
        for it in range(50):
            if time.time() - _START_TIME > _TIMEOUT:
                break
            U, V, W = _als_iteration(U, V, W, n, m, p)
            T = _compute_tensor_fast(U, V, W, n, m, p)
            err = _error(T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _structured_init(rank, n, m, p, seed):
    """Create structured initialization exploiting symmetry."""
    np.random.seed(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Pattern 1: Cycle through indices with some structure
    for r in range(rank):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        # Base values with small random perturbation
        base = 1.0 / np.sqrt(3.0)
        U[i * m + j, r] = base + 0.1 * np.random.randn()
        V[j * p + k, r] = base + 0.1 * np.random.randn()
        W[k * n + i, r] = base + 0.1 * np.random.randn()
    
    # Add some cross-terms for better mixing
    for r in range(rank):
        if r % 3 == 0:
            i2 = (r + 1) % n
            j2 = (r + 1) % m
            k2 = (r + 1) % p
            U[i2 * m + j2, r] += 0.05 * np.random.randn()
            V[j2 * p + k2, r] += 0.05 * np.random.randn()
            W[k2 * n + i2, r] += 0.05 * np.random.randn()
    
    # Normalize columns
    for r in range(rank):
        norm_U = np.linalg.norm(U[:, r])
        norm_V = np.linalg.norm(V[:, r])
        norm_W = np.linalg.norm(W[:, r])
        if norm_U > 1e-10:
            U[:, r] /= norm_U
        if norm_V > 1e-10:
            V[:, r] /= norm_V
        if norm_W > 1e-10:
            W[:, r] /= norm_W
    
    return U, V, W

def _random_init(rank, n, m, p, seed):
    """Create random initialization."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize
    scale = 1.0 / np.sqrt(rank)
    U *= scale
    V *= scale
    W *= scale
    
    return U, V, W

def _try_rank(target_rank, n, m, p, num_restarts=5):
    """Try to find a rank-target_rank decomposition."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        # Alternate between structured and random initialization
        if restart % 2 == 0:
            U, V, W = _structured_init(target_rank, n, m, p, 42 + restart * 1000)
        else:
            U, V, W = _random_init(target_rank, n, m, p, 42 + restart * 1000 + 500)
        
        U_sol, V_sol, W_sol, err = _als_with_polish(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    return best_U, best_V, best_W, best_err

def _high_precision_polish(U, V, W, n, m, p, max_iter=100):
    """High-precision polish pass with very tight tolerance."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor_fast(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor_fast(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-15:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try to find lower rank decompositions
    # For 3x4 * 4x4, theoretical lower bounds suggest rank around 33-40
    # Try ranks from 33 up to 47
    for target_rank in range(33, max_rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        # Determine number of restarts based on remaining time
        remaining = _TIMEOUT - (time.time() - _START_TIME)
        num_restarts = max(3, min(8, int(remaining / 10)))
        
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts=num_restarts)
        
        if err < 1e-10:
            # Found exact decomposition - do high-precision polish
            if time.time() - _START_TIME < _TIMEOUT - 10:
                U_try, V_try, W_try, err = _high_precision_polish(U_try, V_try, W_try, n, m, p)
            
            if err < 1e-10:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                # Try even lower ranks
                continue
        else:
            # If error is small, we might be close - try more restarts
            if err < 0.01 and time.time() - _START_TIME < _TIMEOUT - 15:
                U_try2, V_try2, W_try2, err2 = _try_rank(target_rank, n, m, p, num_restarts=5)
                if err2 < 1e-10:
                    # Do high-precision polish
                    if time.time() - _START_TIME < _TIMEOUT - 10:
                        U_try2, V_try2, W_try2, err2 = _high_precision_polish(U_try2, V_try2, W_try2, n, m, p)
                    if err2 < 1e-10:
                        best_U, best_V, best_W = U_try2, V_try2, W_try2
                        best_rank = target_rank
                        continue
    
    # Convert to lists for return
    U_list = [[float(val) for val in row] for row in best_U.tolist()]
    V_list = [[float(val) for val in row] for row in best_V.tolist()]
    W_list = [[float(val) for val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
