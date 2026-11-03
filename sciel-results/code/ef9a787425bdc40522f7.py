import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition (Rank 24)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Try to find Rank 12
    target_rank = max(n * m, m * p, n * p) # 12
    
    def residual(params):
        U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank+m*p*target_rank], params[n*m*target_rank+m*p*target_rank:]
        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0
                        for r in range(target_rank):
                            val += W_mat[k*n+i, r] * U_mat[i*m+j, r] * V_mat[jp*p+k, r]
                        target = 1.0 if j == jp else 0.0
                        res.append(val - target)
        return np.array(res)

    def snap_and_verify(U, V, W, step):
        U_s = np.round(U / step) * step
        V_s = np.round(V / step) * step
        W_s = np.round(W / step) * step
        res = residual(np.concatenate([U_s.flatten(), V_s.flatten(), W_s.flatten()]))
        if np.max(np.abs(res)) < 1e-6:
            return U_s, V_s, W_s
        return None

    found_rank_12 = False
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(attempt * 100 + 42)
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residual, params0, method='lm', max_nfev=5000)
            
            if result.cost < 1e-8:
                U_mat = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_mat = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_mat = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                
                # Try snapping to half-integers (step 0.5)
                snapped = snap_and_verify(U_mat, V_mat, W_mat, 0.5)
                if snapped:
                    U_s, V_s, W_s = snapped
                    best_result = {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                    best_rank = target_rank
                    found_rank_12 = True
                    break
                else:
                    # Try snapping to integers (step 1.0)
                    snapped = snap_and_verify(U_mat, V_mat, W_mat, 1.0)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        best_result = {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        best_rank = target_rank
                        found_rank_12 = True
                        break
        except Exception:
            continue

    return best_result
