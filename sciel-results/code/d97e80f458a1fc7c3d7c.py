import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * p # 30
    
    # Shapes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target tensor is all ones
    # We need to find U, V, W such that sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
    
    def compute_tensor_flattened(params):
        # params is a flat vector of size (size_U + size_V + size_W) * rank
        rank = target_rank
        U_flat = params[:size_U * rank].reshape(size_U, rank)
        V_flat = params[size_U * rank : size_U * rank + size_V * rank].reshape(size_V, rank)
        W_flat = params[size_U * rank + size_V * rank :].reshape(size_W, rank)
        
        # Compute R[i,j,k]
        # R[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        R = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over t
                    R[i, j, k] = np.sum(U_flat[u_idx, :] * V_flat[v_idx, :] * W_flat[w_idx, :])
        return R.flatten()

    def residuals(params):
        R_flat = compute_tensor_flattened(params)
        return R_flat - 1.0 # Target is 1

    # Initial guess
    np.random.seed(42)
    rank = target_rank
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_size = (size_U + size_V + size_W) * rank
    
    # Try multiple random starts
    best_result = None
    start_time = time.time()
    deadline = start_time + 110.0
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Initialize with small random values
        params = np.random.normal(0, 1, total_size)
        
        # Scale down to avoid overflow initially?
        # The product of 3 terms. If values are ~1, sum of 30 terms is ~30.
        # We want sum=1. So maybe initialize with 0.5?
        # Let's try uniform [0.1, 0.5]
        params = np.random.uniform(0.1, 0.5, total_size)
        
        try:
            res = least_squares(residuals, params, method='lm', max_nfev=10000)
            if res.cost < 1e-10:
                # Check time
                if time.time() > deadline:
                    break
                
                # Extract U, V, W
                U = params[:size_U * rank].reshape(size_U, rank)
                V = params[size_U * rank : size_U * rank + size_V * rank].reshape(size_V, rank)
                W = params[size_U * rank + size_V * rank :].reshape(size_W, rank)
                
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                U_check = U_snapped.flatten()
                V_check = V_snapped.flatten()
                W_check = W_snapped.flatten()
                params_check = np.concatenate([U_check, V_check, W_check])
                
                err = np.max(np.abs(residuals(params_check)))
                if err < 1e-6:
                    # Convert to lists of floats (or ints if they are integers)
                    U_list = U_snapped.tolist()
                    V_list = V_snapped.tolist()
                    W_list = W_snapped.tolist()
                    
                    # Ensure they are represented as numbers, not numpy types
                    U_list = [[float(x) for x in row] for row in U_list]
                    V_list = [[float(x) for x in row] for row in V_list]
                    W_list = [[float(x) for x in row] for row in W_list]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            continue
            
    # Fallback to schoolbook if search fails
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
