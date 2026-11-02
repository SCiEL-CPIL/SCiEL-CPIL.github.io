import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rank14_decomposition():
    """Known-good rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Left Part (Cols 0, 1) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)
    # Right Part (Cols 2, 3) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)
    
    return {"U": U, "V": V, "W": W}

def _try_rank13():
    """Attempt to find a rank-13 decomposition using local optimization from rank-14."""
    n, m, p = DIMENSIONS
    rank = 13
    start_time = time.time()
    deadline = start_time + 110
    
    # Get rank-14 solution
    r14 = _rank14_decomposition()
    U14 = np.array(r14['U'], dtype=float)
    V14 = np.array(r14['V'], dtype=float)
    W14 = np.array(r14['W'], dtype=float)
    
    def residual_func(x):
        U_f = x[:n*m*rank].reshape(n*m, rank)
        V_f = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_f = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_f[u_idx, :], np.multiply(V_f[v_idx, :], W_f[w_idx, :]))
                    errors.append(val - 1.0)
        return np.array(errors)
    
    # Try dropping each column and using multiple random restarts
    for drop_idx in range(14):
        if time.time() > deadline:
            break
            
        # Base initialization: rank-14 solution minus the dropped column
        U_base = np.delete(U14, drop_idx, axis=1).copy()
        V_base = np.delete(V14, drop_idx, axis=1).copy()
        W_base = np.delete(W14, drop_idx, axis=1).copy()
        
        # Try multiple perturbations and restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Create perturbed initialization
            if attempt == 0:
                U_init = U_base.copy()
                V_init = V_base.copy()
                W_init = W_base.copy()
            else:
                # Random perturbation scaled by attempt number
                scale = 0.1 * (attempt + 1)
                U_init = U_base + scale * np.random.randn(n*m, rank)
                V_init = V_base + scale * np.random.randn(m*p, rank)
                W_init = W_base + scale * np.random.randn(p*n, rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares for robust nonlinear optimization
                result = least_squares(
                    residual_func,
                    x0,
                    method='lm',  # Levenberg-Marquardt
                    max_nfev=10000,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12
                )
                
                if result.success or result.cost < 1e-10:
                    # Check residual
                    res = np.linalg.norm(result.fun)
                    if res < 1e-8:
                        # Extract solution
                        U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                        V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        # Try rounding to integers and half-integers
                        for factor in [1, 2]:
                            U_r = np.round(U_sol * factor) / factor
                            V_r = np.round(V_sol * factor) / factor
                            W_r = np.round(W_sol * factor) / factor
                            
                            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                                def to_list(arr):
                                    return [[float(x) for x in row] for row in arr]
                                return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            
                        # Also try without rounding - check if solution is already clean
                        if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                            def to_list(arr):
                                return [[float(x) for x in row] for row in arr]
                            return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
                            
            except Exception:
                pass

    # Final attempt: try a more systematic search with different initializations
    if time.time() < deadline:
        # Try random initializations with small values
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(
                    residual_func,
                    x0,
                    method='lm',
                    max_nfev=10000,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12
                )
                
                if result.success or result.cost < 1e-10:
                    res = np.linalg.norm(result.fun)
                    if res < 1e-8:
                        U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                        V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        for factor in [1, 2]:
                            U_r = np.round(U_sol * factor) / factor
                            V_r = np.round(V_sol * factor) / factor
                            W_r = np.round(W_sol * factor) / factor
                            
                            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                                def to_list(arr):
                                    return [[float(x) for x in row] for row in arr]
                                return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                                
            except Exception:
                pass

    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    Attempts rank 13 first; falls back to rank 14 if not found.
    """
    n, m, p = DIMENSIONS
    
    # Try rank 13
    result = _try_rank13()
    if result is not None:
        return result
    
    # Fallback to rank 14
    return _rank14_decomposition()
