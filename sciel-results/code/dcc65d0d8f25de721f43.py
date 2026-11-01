"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U_f[i * m + j])):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != 1:
                    return False
    return True

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    r = round(2 * x) / 2
    if abs(x - r) < tol:
        return r
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_with_params(rank, n, m, p, deadline):
    """
    Try to find a rank-`rank` decomposition using a structured parameterization.
    
    Ansatz: We split the rank into groups. For the 2x3x10 problem, we exploit
    the structure: U has 6 rows, V has 30 rows, W has 20 rows.
    
    Strategy: Use a block-circulant or shared-parameter structure.
    Specifically, try to find a decomposition where the W matrix has a
    periodic structure in k (the output column index), and U,V are shared.
    
    We parameterize:
    - U: 6 x rank, free parameters
    - V: 30 x rank, free parameters  
    - W: 20 x rank, with structure W[k*n+i] = W[(k mod period)*n + i] for some period
    
    Actually, let's try a more direct approach: use scipy to solve the
    quadratic system directly with random restarts.
    """
    from scipy.optimize import least_squares
    
    start = time.time()
    
    # Total number of scalar variables: 6*rank + 30*rank + 20*rank = 56*rank
    n_vars = (n*m + m*p + p*n) * rank
    n_eqs = n * m * p  # 60 equations
    
    def residuals(params):
        """Compute residuals for all n*m*p tensor entries."""
        U = params[0:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = params[(n*m+m*p)*rank:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    # Tensor entry: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res[idx] = np.dot(u_row, v_row * w_row) - 1.0
                    idx += 1
        return res
    
    # Try multiple random initializations
    best_rank_found = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        np.random.seed(attempt * 12345 + 678)
        x0 = np.random.randn(n_vars) * 0.5
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, 
                                   xtol=1e-12, ftol=1e-12)
            
            if result.cost < 1e-10:
                # Check if we can round to half-integers
                params = result.x
                U = params[0:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
                W = params[(n*m+m*p)*rank:].reshape(p*n, rank)
                
                # Try rounding to half-integers
                U_r = np.zeros_like(U)
                V_r = np.zeros_like(V)
                W_r = np.zeros_like(W)
                success = True
                for idx in range(U.size):
                    r = _round_to_halfint(U.flat[idx])
                    if r is None:
                        success = False
                        break
                    U.flat[idx] = r
                if not success:
                    continue
                for idx in range(V.size):
                    r = _round_to_halfint(V.flat[idx])
                    if r is None:
                        success = False
                        break
                    V.flat[idx] = r
                if not success:
                    continue
                for idx in range(W.size):
                    r = _round_to_halfint(W.flat[idx])
                    if r is None:
                        success = False
                        break
                    W.flat[idx] = r
                if not success:
                    continue
                
                # Verify exact
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_exact(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 60
    
    # Try decreasing ranks from 55 down
    # Known: rank(2x3x1) = 5, so rank(2x3x10) <= 5*10 = 50 if independent
    # But there might be cross-term savings. Try 55, 54, 53, 52, 51, 50
    for rank in range(55, 49, -1):
        if time.time() > deadline:
            break
        result = _try_rank_with_params(rank, n, m, p, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
