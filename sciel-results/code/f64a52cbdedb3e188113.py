import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a lower rank
    # We try R from 83 down to 70 (limited range)
    for R in range(schoolbook_rank - 1, max(70, schoolbook_rank - 15), -1):
        if time.time() > deadline:
            break
            
        # Define the function to minimize: error in tensor reconstruction
        def objective(params):
            # params is a flattened vector of U, V, W
            # Size: (n*m + m*p + p*n) * R
            size_u = n * m * R
            size_v = m * p * R
            U = params[:size_u].reshape(n * m, R)
            V = params[size_u:size_u + size_v].reshape(m * p, R)
            W = params[size_u + size_v:].reshape(p * n, R)
            
            # Compute tensor
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is expensive to compute naively.
            # We can use einsum or loops.
            # For small sizes, loops are fine.
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        errors.append(val - 1)
            return np.array(errors)
            
        # Initial guess: random small integers
        np.random.seed(42)
        p0 = np.random.randint(-1, 2, size=(n * m + m * p + p * n) * R)
        
        # Run optimization
        try:
            res = least_squares(objective, p0, method='lm', max_nfev=1000)
            if res.cost < 1e-6:
                # Extract U, V, W
                size_u = n * m * R
                size_v = m * p * R
                U = res.x[:size_u].reshape(n * m, R)
                V = res.x[size_u:size_u + size_v].reshape(m * p, R)
                W = res.x[size_u + size_v:].reshape(p * n, R)
                
                # Round to nearest half-integer
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_U, best_V, best_W, best_rank = U_int, V_int, W_int, R
                    break # Found a valid lower rank, stop searching lower
        except Exception:
            pass

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
