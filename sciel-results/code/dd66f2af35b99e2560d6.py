import time
import numpy as np
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 135.0

def _check_time():
    return time.time() > _DEADLINE

def _validate_exact(U, V, W, n, m, p):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _validate_float(U, V, W, n, m, p, tol=1e-9):
    """Quick float validation."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, rng, max_als_iters=50):
    """Try to find a rank-target_rank decomposition via ALS from random init."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Random initialization
    U_cur = rng.standard_normal((n * m, target_rank)) * 0.3
    V_cur = rng.standard_normal((m * p, target_rank)) * 0.3
    W_cur = rng.standard_normal((p * n, target_rank)) * 0.3
    
    for it in range(max_als_iters):
        if _check_time():
            return None
        
        # Solve for U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                b = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_cur[i * m + j, :] = sol
                except:
                    pass
        
        # Solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_cur[j * p + k, :] = sol
                except:
                    pass
        
        # Solve for W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                b = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_cur[k * n + i, :] = sol
                except:
                    pass
        
        if it % 10 == 0 and _validate_float(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
            # Try rationalization
            result = _rationalize(U_cur, V_cur, W_cur, n, m, p)
            if result is not None:
                return result
            break
    
    return None

def _rationalize(U, V, W, n, m, p, max_denom=8):
    """Convert float decomposition to exact rational entries."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    if not _validate_float(U_arr, V_arr, W_arr, n, m, p, tol=1e-6):
        return None
    
    def to_frac(x, max_d=max_denom):
        # Try to find best rational approximation
        best = None
        best_err = float('inf')
        for d in range(1, max_d + 1):
            num = round(x * d)
            f = Fraction(num, d)
            err = abs(float(f) - x)
            if err < best_err:
                best_err = err
                best = f
        if best_err > 1e-5:
            return None
        return best
    
    U_fr = []
    for row in U_arr:
        fr_row = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return None
            fr_row.append(f)
        U_fr.append(fr_row)
    
    V_fr = []
    for row in V_arr:
        fr_row = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return None
            fr_row.append(f)
        V_fr.append(fr_row)
    
    W_fr = []
    for row in W_arr:
        fr_row = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return None
            fr_row.append(f)
        W_fr.append(fr_row)
    
    if _validate_exact(U_fr, V_fr, W_fr, n, m, p):
        return U_fr, V_fr, W_fr
    return None

def construct():
    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 48
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 24 up to 47
    for target_rank in range(24, best_rank):
        if _check_time():
            break
        
        found = False
        for restart in range(5):
            if _check_time():
                break
            
            result = _try_reduce_rank(U_school, V_school, W_school, n, m, p, target_rank, rng)
            if result is not None:
                U_fr, V_fr, W_fr = result
                best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, continue to try lower
            pass
    
    return best_result
