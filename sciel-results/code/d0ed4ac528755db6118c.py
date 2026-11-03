import time
import numpy as np
import scipy.optimize as so

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _residuals(params, n, m, p, r):
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:size_U + size_V + size_W]
    
    U = U_flat.reshape((n * m, r))
    V = V_flat.reshape((m * p, r))
    W = W_flat.reshape((p * n, r))
    
    residuals = []
    
    # Target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                residuals.append(val - 1.0)
                
    # Non-target entries
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    residuals.append(val)
                    
    return np.array(residuals)

def _try_rank(n, m, p, r, max_restarts=5):
    if _time_left() < 20:
        return None
        
    n_vars = (n * m + m * p + p * n) * r
    
    best_sol = None
    best_cost = float('inf')
    
    for restart in range(max_restarts):
        if _time_left() < 10:
            break
            
        np.random.seed(42 + restart)
        x0 = np.random.randn(n_vars) * 0.1
        
        try:
            # Adjust max_nfev based on time
            # 10000 evaluations is a reasonable guess
            res = so.least_squares(_residuals, x0, args=(n, m, p, r), method='lm', max_nfev=5000)
            cost = res.cost
            if cost < best_cost:
                best_cost = cost
                best_sol = res.x
                
                if cost < 1e-10:
                    break
        except Exception:
            pass
            
    if best_sol is None:
        return None
        
    if best_cost > 1e-6:
        return None
        
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = best_sol[:size_U].reshape((n * m, r))
    V = best_sol[size_U:size_U + size_V].reshape((m * p, r))
    W = best_sol[size_U + size_V:size_U + size_V + size_W].reshape((p * n, r))
    
    # Snap to half-integers
    def snap(arr):
        return np.round(arr * 2) / 2.0
        
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-9):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    # Try integers
    def snap_int(arr):
        return np.round(arr)
        
    U_i = snap_int(U)
    V_i = snap_int(V)
    W_i = snap_int(W)
    
    if _verify(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p, tol=1e-9):
        return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
        
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # Try ranks
    for r in [18, 19, 20, 21, 22, 23, 24, 27, 30, 33]:
        if _time_left() < 15:
            break
        result = _try_rank(n, m, p, r, max_restarts=3)
        if result is not None:
            return result
            
    return _construct_rank_36(n, m, p)
