import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138  # seconds, leave 12s margin for 150s limit

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0]) if U else 0
    
    def to_int2(x):
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True

def _build_schoolbook(n, m, p):
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

def _solve_for_w(U, V, n, m, p, rank):
    """
    Given U and V, solve for W such that sum_c U_ij_c V_jk_c W_ki_c = 1 for all i,j,k.
    Returns W as a list of lists, or None if no solution found (should not happen for underdetermined).
    """
    # Variables: W[k, i, c] for k in p, i in n, c in rank
    # Total vars: p * n * rank
    num_vars = p * n * rank
    
    # Equations: for each i, j, k
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars), dtype=np.float64)
    b = np.ones(num_eqs, dtype=np.float64)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                # Variable index for W[k, i, c] is (k * n + i) * rank + c
                var_base = (k * n + i) * rank
                for c in range(rank):
                    A[eq_idx, var_base + c] = U[i * m + j, c] * V[j * p + k, c]
                b[eq_idx] = 1
                eq_idx += 1
    
    # Solve A W = b
    # Use lstsq for stability
    try:
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    W = W_flat.reshape(p * n, rank)
    return W

def _search_rank(R, n, m, p, max_trials=1000):
    """
    Search for a decomposition of rank R by fixing U, V randomly and solving for W.
    """
    rng = np.random.default_rng(42)
    
    for t in range(max_trials):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return None
        
        # Random U, V with small integer entries
        # Using {-1, 0, 1} or {-2, -1, 0, 1, 2}
        U = rng.integers(-1, 2, size=(n * m, R))
        V = rng.integers(-1, 2, size=(m * p, R))
        
        W = _solve_for_w(U, V, n, m, p, R)
        if W is None:
            continue
        
        # Try rounding W to integers
        W_int = np.round(W).astype(int)
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_int.tolist()
        
        if _verify_exact(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try rounding W to half-integers
        W_half = np.round(W * 2) / 2
        W_half_list = W_half.tolist()
        
        if _verify_exact(U_list, V_list, W_half_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_half_list}
            
    return None

def construct():
    n, m, p = 2, 4, 7
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks lower than schoolbook
    # Start from 55 down to 40
    for rank in range(55, 39, -1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Allocate more trials for lower ranks? Or just fixed?
        # Let's try 200 trials per rank
        res = _search_rank(rank, n, m, p, max_trials=200)
        if res is not None:
            return res
            
    return best
