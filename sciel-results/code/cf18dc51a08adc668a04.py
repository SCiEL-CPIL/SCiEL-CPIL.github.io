"""Stronger program: Using SciPy Least Squares to find rank-23 decomposition of 3x3x3 matrix multiplication tensor."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_real(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify over reals: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    if isinstance(U, list):
        U = np.array(U)
    if isinstance(V, list):
        V = np.array(V)
    if isinstance(W, list):
        W = np.array(W)
        
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(S, np.ones((n, m, p)), atol=tol)

def _residuals(x, rank, n=3, m=3, p=3):
    """
    Calculate residuals for least squares optimization.
    x is a flat vector of size (n*m + m*p + p*n) * rank.
    """
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:].reshape(p * n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    
    return (S - target).flatten()

def _try_least_squares(rank, n=3, m=3, p=3, max_iters=200, time_budget=10.0, seed=42):
    """
    Try to find a rank-r decomposition using scipy.optimize.least_squares.
    """
    start = time.time()
    np.random.seed(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    best_cost = float('inf')
    best_x = None
    
    for attempt in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Initialize with small random values
        x0 = np.random.randn(total_size) * 0.1
        
        try:
            result = least_squares(
                _residuals, 
                x0, 
                args=(rank, n, m, p),
                method='lm', # Levenberg-Marquardt
                max_nfev=100
            )
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
                
            # Check if converged to a valid solution
            if result.cost < 1e-8:
                U = best_x[:size_U].reshape(n * m, rank)
                V = best_x[size_U:size_U + size_V].reshape(m * p, rank)
                W = best_x[size_U + size_V:].reshape(p * n, rank)
                return U, V, W, True
                
        except Exception:
            continue
            
    if best_x is not None and best_cost < 1e-8:
        U = best_x[:size_U].reshape(n * m, rank)
        V = best_x[size_U:size_U + size_V].reshape(m * p, rank)
        W = best_x[size_U + size_V:].reshape(p * n, rank)
        return U, V, W, True
        
    return None, None, None, False

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p  # 27
    
    # Try to find lower rank decompositions
    # Known: rank 23 is achievable for 3x3x3
    for r in range(23, best_rank):
        if time.time() > deadline:
            break
        
        # Try least squares
        try:
            U_ls, V_ls, W_ls, success = _try_least_squares(
                r, n, m, p, max_iters=20, time_budget=10.0, seed=42+r
            )
            if success:
                # Verify before accepting
                if _verify_real(U_ls, V_ls, W_ls, n, m, p, tol=1e-6):
                    best = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                    best_rank = r
                    break
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    return best
# EVOLVE-BLOCK-END
