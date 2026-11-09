import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find rank 20 decomposition of M_{2,3,4}
    n2, m2, p2 = 2, 3, 4
    rank20 = 20
    
    # Target tensor T2 for M_{2,3,4}
    # T2[i, j, k] = 1
    # Indices: i in 0..1, j in 0..2, k in 0..3
    
    # We want U2 (6x20), V2 (12x20), W2 (8x20)
    
    best_U2 = None
    best_V2 = None
    best_W2 = None
    best_error = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        U2 = np.random.uniform(-1, 1, (n2 * m2, rank20))
        V2 = np.random.uniform(-1, 1, (m2 * p2, rank20))
        W2 = np.random.uniform(-1, 1, (p2 * n2, rank20))
        
        # Flatten parameters
        x0 = np.concatenate([U2.flatten(), V2.flatten(), W2.flatten()])
        
        def residuals(x):
            U = x[:6*20].reshape(6, 20)
            V = x[6*20:6*20+12*20].reshape(12, 20)
            W = x[6*20+12*20:].reshape(8, 20)
            
            # Compute T = sum_c U[:,c] (outer) V[:,c] (outer) W[:,c]
            # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            
            T = np.zeros((2, 3, 4))
            for c in range(20):
                u_c = U[:, c]
                v_c = V[:, c]
                w_c = W[:, c]
                
                # u_c is 6x1, v_c is 12x1, w_c is 8x1
                # We need to add u_c (outer) v_c (outer) w_c (outer) to T
                # This is a rank-1 tensor update
                
                # Unroll loops for speed
                for i in range(2):
                    for j in range(3):
                        u_val = u_c[i*3 + j]
                        if u_val == 0: continue
                        for k in range(4):
                            v_val = v_c[j*4 + k]
                            if v_val == 0: continue
                            w_val = w_c[k*2 + i]
                            T[i, j, k] += u_val * v_val * w_val
                            
            # Residuals: T - 1
            return (T - 1).flatten()
            
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            x_opt = result.x
            U2_opt = x_opt[:6*20].reshape(6, 20)
            V2_opt = x_opt[6*20:6*20+12*20].reshape(12, 20)
            W2_opt = x_opt[6*20+12*20:].reshape(8, 20)
            
            # Check error
            err = np.linalg.norm(residuals(x_opt))
            if err < best_error:
                best_error = err
                best_U2 = U2_opt
                best_V2 = V2_opt
                best_W2 = W2_opt
                
                # Try to round to 0.5
                U2_r = np.round(U2_opt * 2) / 2
                V2_r = np.round(V2_opt * 2) / 2
                W2_r = np.round(W2_opt * 2) / 2
                
                # Check if rounded version is exact
                T_check = np.zeros((2, 3, 4))
                for c in range(20):
                    u_c = U2_r[:, c]
                    v_c = V2_r[:, c]
                    w_c = W2_r[:, c]
                    for i in range(2):
                        for j in range(3):
                            u_val = u_c[i*3 + j]
                            if u_val == 0: continue
                            for k in range(4):
                                v_val = v_c[j*4 + k]
                                if v_val == 0: continue
                                w_val = w_c[k*2 + i]
                                T_check[i, j, k] += u_val * v_val * w_val
                                
                if np.allclose(T_check, 1, atol=1e-9):
                    # Found valid decomposition!
                    # Construct M_{2,3,8} rank 40
                    U = np.concatenate([U2_r, U2_r], axis=1)
                    V = np.concatenate([V2_r, V2_r], axis=1)
                    W = np.zeros((16, 40))
                    # First 20 cols
                    W[0:4, 0:20] = W2_r
                    # Next 20 cols
                    W[4:8, 20:40] = W2_r
                    
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        except Exception:
            pass
            
    # Fallback to schoolbook
    rank = rank_schoolbook
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
