import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin

    n, m, p = DIMENSIONS
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We want to find U, V, W with rank R
    # Try ranks from 62 down to some limit
    # Trivial rank is 63
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Define the objective function
    def objective(params, R):
        # Reshape params into U, V, W
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[0:size_U].reshape(n*m, R)
        V = params[size_U:size_U+size_V].reshape(m*p, R)
        W = params[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
        
        # Compute tensor contraction
        # T_comp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can use einsum
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        err = np.sum((T - T_comp) ** 2)
        return err

    # Try a few ranks
    # We can't try too many due to time limits.
    # Let's try 62, 61, 60.
    
    for R in [62, 61, 60]:
        if time.time() > deadline:
            break
            
        # Random initialization
        # We need to be careful with scale.
        # Initialize with small random values.
        
        found = False
        # Try a few restarts
        for restart in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R + restart)
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            # Initialize
            x0 = np.random.randn(total_size) * 0.1
            
            # Optimization
            # L-BFGS-B is good for this size
            res = minimize(objective, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or res.fun < 1e-8:
                # Verify
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U = res.x[0:size_U].reshape(n*m, R)
                V = res.x[size_U:size_U+size_V].reshape(m*p, R)
                W = res.x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_comp, T, atol=1e-8):
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            break

    if best_U is not None:
        # Return as lists of lists to match interface if needed, or numpy arrays?
        # The prompt says: "factor matrices of shapes ... with integer/half-integer entries"
        # But the current program returns lists of lists of floats.
        # "return {'U', 'V', 'W'}: factor matrices"
        # The current program returns lists.
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to trivial decomposition
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
