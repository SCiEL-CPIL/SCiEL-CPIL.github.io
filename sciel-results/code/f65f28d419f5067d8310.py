import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank(r, n_attempts=10):
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r
        total_size = U_size + V_size + W_size
        
        def objective(params):
            U_flat = params[:U_size].reshape(n*m, r)
            V_flat = params[U_size:U_size+V_size].reshape(m*p, r)
            W_flat = params[U_size+V_size:U_size+V_size+W_size].reshape(p*n, r)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        err += (val - 1.0)**2
            return err

        for attempt in range(n_attempts):
            if time.time() > deadline - 25:
                return None
            
            np.random.seed(attempt * 7919 + r * 13)
            # Initialize with small random values
            params0 = np.random.uniform(-0.5, 0.5, total_size)
            
            try:
                res = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e-10:
                    U = res.x[:U_size].reshape(n*m, r)
                    V = res.x[U_size:U_size+V_size].reshape(m*p, r)
                    W = res.x[U_size+V_size:U_size+V_size+W_size].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), 1e-5):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            except Exception:
                continue
        
        return None

    # Try ranks from 24 upwards
    # Known: 36 (schoolbook), 32 (23+9), maybe lower?
    for r in range(24, 33):
        if time.time() > deadline - 30:
            break
        result = try_rank(r)
        if result is not None:
            return result
    
    return schoolbook()
