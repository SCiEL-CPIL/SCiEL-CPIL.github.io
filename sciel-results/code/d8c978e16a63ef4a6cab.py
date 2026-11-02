"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _target_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D array T[a,b,c] where
    a = i*m+j (U index), b = j*p+k (V index), c = k*n+i (W index).
    T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _decompose_from_factors(U, V, W):
    """Given factor matrices, return the reconstructed tensor."""
    rank = U.shape[1]
    # T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly matches the target tensor."""
    T_target = _target_tensor(n, m, p)
    T_recon = _decompose_from_factors(U, V, W)
    return np.max(np.abs(T_target - T_recon)) < tol

def _try_rank(n, m, p, rank, n_trials=50, time_limit=12.0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    T_target = _target_tensor(n, m, p)
    T_flat = T_target.reshape(-1)
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    def objective(params):
        # params: [U_flat (dim_u*rank), V_flat (dim_v*rank), W_flat (dim_w*rank)]
        u = dim_u * rank
        v = u + dim_v * rank
        U = params[:u].reshape(dim_u, rank)
        V = params[u:v].reshape(dim_v, rank)
        W = params[v:].reshape(dim_w, rank)
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for c in range(rank):
            T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        diff = T.reshape(-1) - T_flat
        return np.sum(diff * diff)
    
    best_obj = np.inf
    best_params = None
    start = time.time()
    
    for trial in range(n_trials):
        if time.time() - start > time_limit:
            break
        # Random initialization
        scale = 1.0 / (rank ** 0.5)
        params0 = np.random.randn(dim_u * rank + dim_v * rank + dim_w * rank) * scale
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                if best_obj < 1e-12:
                    break
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    u = dim_u * rank
    v = u + dim_v * rank
    U = best_params[:u].reshape(dim_u, rank)
    V = best_params[u:v].reshape(dim_v, rank)
    W = best_params[v:].reshape(dim_w, rank)
    
    # Round to nearest half-integer and verify
    U_round = np.round(U * 2.0) / 2.0
    V_round = np.round(V * 2.0) / 2.0
    W_round = np.round(W * 2.0) / 2.0
    
    if _verify_exact(U_round, V_round, W_round, n, m, p, tol=1e-9):
        return (U_round, V_round, W_round)
    
    # Also try rounding to integers
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify_exact(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return (U_int, V_int, W_int)
    
    return None

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 10s safety margin
    
    # Known lower bound: max(n*m, m*p, n*p)
    lower_bound = max(n*m, m*p, n*p)
    schoolbook_rank = n * m * p
    
    # Try ranks from lower_bound up to schoolbook_rank
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, rank, n_trials=30, time_limit=10.0)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
