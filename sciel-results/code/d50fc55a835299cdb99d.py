import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 10
    
    # Dimensions of U, V, W
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    # Target Tensor M
    M = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m + j, j*p + k, k*n + i] = 1.0

    def residual(theta):
        # Unpack theta into U, V, W
        # U: 4x10, V: 6x10, W: 6x10
        # Total size: 40 + 60 + 60 = 160
        U = theta[0:40].reshape(4, 10)
        V = theta[40:100].reshape(6, 10)
        W = theta[100:160].reshape(6, 10)
        
        # Compute T
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual
        return (T - M).flatten()

    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_error = float('inf')
    best_theta = None
    
    # Try multiple restarts
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        theta0 = np.random.randn(160) * 0.5
        
        # Use least_squares
        try:
            res = opt.least_squares(residual, theta0, method='lm', max_nfev=1000)
            err = res.cost
            if err < best_error:
                best_error = err
                best_theta = res.x
                
                # Check if close to integers/half-integers
                U = best_theta[0:40].reshape(4, 10)
                V = best_theta[40:100].reshape(6, 10)
                W = best_theta[100:160].reshape(6, 10)
                
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                T_snapped = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                err_snapped = np.linalg.norm(T_snapped - M)
                
                if err_snapped < 1e-6:
                    # Found a valid integer/half-integer solution
                    return {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
        except Exception:
            continue

    # If no exact solution found, return the best numerical one rounded, or schoolbook
    # The task requires exact decomposition. If we don't find one, we should probably return schoolbook to be safe, 
    # but the score is rank. Rank 12 is valid. Rank 10 is better.
    # If we found a low error but not exact, it's invalid.
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
