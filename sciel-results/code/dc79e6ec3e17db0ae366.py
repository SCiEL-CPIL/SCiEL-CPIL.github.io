import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 3
    
    def compute_tensor_flat(U, V, W, R):
        T = np.zeros(n * m * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[(i*m+j)*m*p + (j*p+k)*n + (k*n+l)] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+l])
        return T

    # Target tensor
    T_target = np.zeros(n * m * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    if j == k:
                        T_target[(i*m+j)*m*p + (j*p+k)*n + (k*n+l)] = 1.0

    def residual(params, R):
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R : n*m*R + m*p*R + p*n*R].reshape(p*n, R)
        T = compute_tensor_flat(U, V, W, R)
        return T - T_target

    def try_rank(R):
        for seed in range(10):
            np.random.seed(seed)
            # Initialize
            U0 = np.random.randn(n*m, R) * 0.5
            V0 = np.random.randn(m*p, R) * 0.5
            W0 = np.random.randn(p*n, R) * 0.5
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if result.success and result.cost < 1e-10:
                U = result.x[:n*m*R].reshape(n*m, R)
                V = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = result.x[n*m*R + m*p*R : n*m*R + m*p*R + p*n*R].reshape(p*n, R)
                
                # Try to snap to integers
                all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                # Find best scale
                best_scale = None
                best_err = float('inf')
                for scale in [1, 2, 4, 8, 16, 32]:
                    scaled = all_vals * scale
                    rounded = np.round(scaled)
                    err = np.max(np.abs(rounded - scaled))
                    if err < 1e-5 and err < best_err:
                        best_err = err
                        best_scale = scale
                        
                if best_scale is not None:
                    U_int = np.round(U * best_scale)
                    V_int = np.round(V * best_scale)
                    W_int = np.round(W * best_scale)
                    
                    # Verify
                    U_f = U_int.astype(float)
                    V_f = V_int.astype(float)
                    W_f = W_int.astype(float)
                    T_check = compute_tensor_flat(U_f, V_f, W_f, R)
                    if np.max(np.abs(T_check - T_target)) < 1e-8:
                        return {
                            "U": [[int(x) for x in row] for row in U_int],
                            "V": [[int(x) for x in row] for row in V_int],
                            "W": [[int(x) for x in row] for row in W_int]
                        }
        return None

    for R in [8, 9, 10]:
        sol = try_rank(R)
        if sol:
            return sol
            
    # Fallback rank 12
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for l in range(3):
                U[i*2+j][col] = 1
                V[j*3+l][col] = 1
                W[l*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
