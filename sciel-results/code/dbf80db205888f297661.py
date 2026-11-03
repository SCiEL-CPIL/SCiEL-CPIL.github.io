"""Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]
DEADLINE = 110.0  # safety margin before 120s limit
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_remaining():
    return max(0.0, DEADLINE - _elapsed())

def _build_target_tensor():
    """Build the matrix multiplication tensor T[a,b,c] for 2x3x3."""
    n, m, p = DIMENSIONS
    # a = i*m + k (i in 0..n-1, k in 0..m-1) -> 0..5
    # b = k*p + j (k in 0..m-1, j in 0..p-1) -> 0..8
    # c = i*p + j (i in 0..n-1, j in 0..p-1) -> 0..5
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                target[a, b, c] = 1.0
    return target

def _residuals(x, target_flat, n, m, p, R):
    """Compute residuals for least_squares."""
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    U = x[:n_rows_U * R].reshape(n_rows_U, R)
    V = x[n_rows_U * R:n_rows_U * R + n_rows_V * R].reshape(n_rows_V, R)
    W = x[n_rows_U * R + n_rows_V * R:].reshape(n_rows_W, R)
    # Compute tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return T.ravel() - target_flat

def _try_rank_schoolbook():
    """Return the rank-18 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    target = _build_target_tensor()
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _round_to_half_integer(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0

def _polish_and_verify(U, V, W, R):
    """Try rounding to half-integers and verify."""
    U_r = _round_to_half_integer(U)
    V_r = _round_to_half_integer(V)
    W_r = _round_to_half_integer(W)
    if _verify_exact(U_r, V_r, W_r):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    target = _build_target_tensor()
    target_flat = target.ravel()
    
    # Try rank 17 with numerical search
    for R in [17, 16, 15, 14]:
        if _time_remaining() < 5.0:
            break
        n_vars = (n*m + m*p + p*n) * R
        max_restarts = 50
        found = None
        for restart in range(max_restarts):
            if _time_remaining() < 3.0:
                break
            # Random initialization
            np.random.seed(42 + restart * 1000)
            x0 = np.random.randn(n_vars) * 0.5
            # Scale: for rank 17, entries are typically small integers
            x0 *= (2.0 / np.sqrt(R))
            
            try:
                result = least_squares(
                    _residuals, x0, args=(target_flat, n, m, p, R),
                    method='lm', max_nfev=20000
                )
            except Exception:
                continue
            
            if result.success or result.cost < 1e-6:
                U_flat = result.x[:n*m*R].reshape(n*m, R)
                V_flat = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_flat = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Try rounding
                polished = _polish_and_verify(U_flat, V_flat, W_flat, R)
                if polished is not None:
                    found = polished
                    break
                
                # Also try direct verification of the numerical solution
                if _verify_exact(U_flat, V_flat, W_flat):
                    found = (U_flat.tolist(), V_flat.tolist(), W_flat.tolist())
                    break
        
        if found is not None:
            return {"U": found[0], "V": found[1], "W": found[2]}
    
    # Fallback to schoolbook rank 18
    return _try_rank_schoolbook()
