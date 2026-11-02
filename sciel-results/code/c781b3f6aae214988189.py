"""
Rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
"""
import numpy as np
import time
import itertools
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
DEADLINE = 150  # seconds

def _remaining_time():
    return max(0, DEADLINE - (time.time() - START_TIME))

def _to_fraction(x):
    """Convert numpy/scalar to Fraction."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    if isinstance(x, (float, np.floating)):
        # Try to express as a simple fraction
        return Fraction(x).limit_denominator(1000)
    return Fraction(0)

def _fracs_equal(a, b, tol=1e-12):
    return abs(a - b) <= tol

def _fracs_to_float(f):
    if isinstance(f, Fraction):
        return float(f)
    return float(f)

def _build_tensor(n, m, p):
    """
    Build the target tensor as a dense numpy array.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _decompose_search(n, m, p, target_rank, max_restarts=50, time_budget=10):
    """
    Search for a decomposition of the given rank.
    Returns U, V, W as numpy arrays or None.
    """
    T_target = _build_tensor(n, m, p)
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    total_params = target_rank * (size_u + size_v + size_w)
    
    def objective(x):
        U = x[:target_rank * size_u].reshape(target_rank, size_u)
        V = x[target_rank * size_u : target_rank * (size_u + size_v)].reshape(target_rank, size_v)
        W = x[target_rank * (size_u + size_v):].reshape(target_rank, size_w)
        
        T_rec = np.einsum('ra, rb, rc -> abc', U, V, W)
        diff = T_rec - T_target
        return np.sum(diff * diff)

    best_error = float('inf')
    best_x = None
    
    start_time = time.time()
    for i in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        if _remaining_time() < 10:
            break
            
        x0 = np.random.uniform(-1, 1, total_params)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxls': 50})
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
                if best_error < 1e-10:
                    break
        except Exception:
            continue
            
    if best_x is None or best_error > 1e-6:
        return None
        
    U = best_x[:target_rank * size_u].reshape(target_rank, size_u)
    V = best_x[target_rank * size_u : target_rank * (size_u + size_v)].reshape(target_rank, size_v)
    W = best_x[target_rank * (size_u + size_v):].reshape(target_rank, size_w)
    
    def try_round(arr):
        arr_int = np.round(arr)
        arr_half = np.round(arr * 2) / 2.0
        
        err_int = np.sum((arr_int - arr)**2)
        err_half = np.sum((arr_half - arr)**2)
        
        if err_int < 1e-6:
            return arr_int
        elif err_half < 1e-6:
            return arr_half
        else:
            if err_int < 1e-4:
                return arr_int
            if err_half < 1e-4:
                return arr_half
            return None

    U_r = try_round(U)
    V_r = try_round(V)
    W_r = try_round(W)
    
    if U_r is None or V_r is None or W_r is None:
        return None
        
    T_rec = np.einsum('ra, rb, rc -> abc', U_r, V_r, W_r)
    if np.allclose(T_rec, T_target, atol=1e-9):
        return U_r, V_r, W_r
        
    return None





def construct():
    """
    Find a low-rank tensor decomposition for 3x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try to find a decomposition with lower rank
    # We prioritize lower ranks.
    for rank in range(17, best_rank + 1):
        if _remaining_time() < 20:
            break
            
        # Allocate time budget for this rank
        # If we find a solution, we stop.
        res = _decompose_search(n, m, p, rank, max_restarts=100, time_budget=min(10, _remaining_time() - 5))
        
        if res is not None:
            U, V, W = res
            best_rank = rank
            best_decomp = (U, V, W)
            break # Found the minimal rank (assuming 17 is the minimum)
            
    if best_decomp is None:
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=int)
        V = np.zeros((m*p, n*m*p), dtype=int)
        W = np.zeros((p*n, n*m*p), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
        best_decomp = (U, V, W)
        best_rank = n * m * p
        
    U, V, W = best_decomp
    
    # Convert to list of lists of floats/ints
    # The task asks for integer/half-integer entries.
    # If they are integers, return int. If half-integers, return float.
    
    def to_list(arr):
        out = []
        for row in arr:
            row_list = []
            for val in row:
                if val == int(val):
                    row_list.append(int(val))
                else:
                    row_list.append(float(val))
            out.append(row_list)
        return out
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
