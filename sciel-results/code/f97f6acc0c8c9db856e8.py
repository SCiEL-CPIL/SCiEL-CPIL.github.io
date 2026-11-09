import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Pre-compute target tensor
    # Shape: (n, m, m, p, p, n)
    # T[i, j, j2, k2, k1, i2] = 1 if i==i2 and j==j2 and k1==k2 else 0
    T = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0
                
    for R in range(20, 31):
        if time.time() > deadline:
            break
            
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            # Initialize
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_ = x[:n*m*R].reshape(n*m, R)
                V_ = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_ = x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Reshape to 3D
                U3 = U_.reshape(n, m, R)
                V3 = V_.reshape(m, p, R)
                W3 = W_.reshape(p, n, R)
                
                # Compute S
                # S[i, j, j2, k2, k1, i2] = sum_r U[i,j,r] V[j2,k2,r] W[k1,i2,r]
                S = np.einsum('ijr,j2k2r,k1i2r->ijj2k2k1i2', U3, V3, W3)
                
                err = np.sum((S - T)**2)
                return err
                
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
            
            if res.fun < 1e-6:
                U_sol = res.x[:n*m*R].reshape(n*m, R)
                V_sol = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_sol = res.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Try rounding to half-integers
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Check error
                U3 = U_int.reshape(n, m, R)
                V3 = V_int.reshape(m, p, R)
                W3 = W_int.reshape(p, n, R)
                S = np.einsum('ijr,j2k2r,k1i2r->ijj2k2k1i2', U3, V3, W3)
                err_int = np.sum((S - T)**2)
                
                if err_int < 1e-6:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
    # Fallback
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
