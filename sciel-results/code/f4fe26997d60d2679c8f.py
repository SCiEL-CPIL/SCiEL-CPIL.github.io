"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    # Check exact start time
    deadline = time.time() + 145  # Reserve 5s margin for 150s total
    
    # Attempt numerical search for lower rank if sufficient time
    # Known bounds: Lower bound 20, Upper bound 34 (current seed)
    # Try ranks from 20 to 33
    for target_rank in range(20, 34):
        if time.time() >= deadline:
            break
        
        res = attempt_lower_rank(target_rank, deadline)
        if res is not None and valid_res(res, 3, 3, 4):
            return res
    
    # Fallback: return exactly rank 34 decomposition
    return construct_r34()

def attempt_lower_rank(target_rank, deadline):
    n, m, p = 3, 3, 4
    R = target_rank
    
    # Precompute indices for the tensor T
    # T[a, b, c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Total entries for full verification
    all_a = np.repeat(np.arange(n*m), m*p)
    all_b = np.tile(np.arange(m*p), n*m)
    all_c = np.tile(np.arange(p*n), n*m * m*p)
    
    # Create full target tensor vector
    T_full = np.zeros(n*m * m*p * p*n)
    # Flatten indices for support
    # We need to map (a,b,c) to a flat index
    # idx = a * (m*p * p*n) + b * (p*n) + c
    for idx in range(len(indices_a)):
        a = indices_a[idx]
        b = indices_b[idx]
        c = indices_c[idx]
        flat_idx = a * (m*p * p*n) + b * (p*n) + c
        T_full[flat_idx] = 1.0
        
    # Objective function
    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Compute sum_r U[a,r] V[b,r] W[c,r] for all a,b,c
        # This is a tensor contraction.
        # For efficiency, we can compute the full tensor product?
        # U (nm x R), V (mp x R), W (pn x R)
        # Result T (nm x mp x pn)
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Using einsum might be fast
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a full array of size 27 * 12 * 12 = 3888
        # This is small, so it's fine.
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T_calc.flatten()
        
        error = np.sum((T_flat - T_full)**2)
        return error

    # Initial guess
    # Try a few random restarts
    best_res = None
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
            
        np.random.seed(42 + restart)
        # Scale initial values to avoid large gradients
        # Typical values in decompositions are small integers
        scale = 1.0
        params = np.random.randn((n*m + m*p + p*n) * R) * scale
        
        # Optimization
        try:
            from scipy.optimize import minimize
            res_opt = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
            
            if res_opt.fun < 1e-6:
                U = res_opt.x[0:n*m*R].reshape(n*m, R)
                V = res_opt.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res_opt.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try rounding to nearest 0.5
                # Also try nearest integer
                for step in [0.5, 1.0]:
                    U_rounded = np.round(U / step) * step
                    V_rounded = np.round(V / step) * step
                    W_rounded = np.round(W / step) * step
                    
                    # Verify exactness
                    T_calc = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    T_flat = T_calc.flatten()
                    
                    if np.allclose(T_flat, T_full, atol=1e-9):
                        # Convert to lists of numbers
                        # Ensure they are floats or ints as required? "integer/half-integer"
                        # Python floats are fine if they are exact
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
        except Exception:
            pass
            
    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
        
    # Build full target tensor
    T_full = np.zeros(n*m * m*p * p*n)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                flat_idx = a * (m*p * p*n) + b * (p*n) + c
                T_full[flat_idx] = 1.0
                
    # Compute candidate tensor
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    T_flat = T_calc.flatten()
    
    return np.allclose(T_flat, T_full, atol=1e-9)

def construct_r34():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
