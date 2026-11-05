import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find rank 23 decomposition
    target_rank = 23
    if target_rank < rank_sb:
        try:
            # Setup variables
            # U: (n*m, r), V: (m*p, r), W: (p*n, r)
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            total_size = size_U + size_V + size_W
            
            # Residual function
            def residuals(x):
                U = x[:size_U].reshape((n * m, target_rank))
                V = x[size_U:size_U + size_V].reshape((m * p, target_rank))
                W = x[size_U + size_V:].reshape((p * n, target_rank))
                
                # Compute tensor T_c = sum_c U_c V_c W_c
                # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We want T[i, j, k] == 1 for all i, j, k
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i * m + j]
                            v_val = V[j * p + k]
                            w_val = W[k * n + i]
                            # Dot product of u_val, v_val, w_val?
                            # No, it's sum_c u_val[c] * v_val[c] * w_val[c]
                            res[i, j, k] = np.sum(u_val * v_val * w_val)
                # Target is 1 for all entries
                return (res - 1).flatten()
            
            # Initial guess: Random
            # Or use schoolbook projected?
            # Let's try random initialization
            np.random.seed(42)
            x0 = np.random.randn(total_size) * 0.1
            
            # Solve
            start_time = time.time()
            timeout = 100 # seconds
            
            # Use least_squares
            from scipy.optimize import least_squares
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            
            if time.time() - start_time > timeout:
                return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
                
            # Check if solution is good
            if result.success or result.cost < 1e-10:
                # Extract U, V, W
                U_sol = result.x[:size_U].reshape((n * m, target_rank))
                V_sol = result.x[size_U:size_U + size_V].reshape((m * p, target_rank))
                W_sol = result.x[size_U + size_V:].reshape((p * n, target_rank))
                
                # Try to snap to half-integers
                # Check if values are close to multiples of 0.5
                def snap_to_half_int(matrix):
                    snapped = np.zeros_like(matrix)
                    valid = True
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            val = matrix[i, j]
                            # Check if close to k/2
                            k = round(val * 2)
                            snapped_val = k / 2.0
                            if abs(val - snapped_val) > 1e-6:
                                valid = False
                                break
                            snapped[i, j] = snapped_val
                        if not valid:
                            break
                    return snapped if valid else None
                
                U_snap = snap_to_half_int(U_sol)
                V_snap = snap_to_half_int(V_sol)
                W_snap = snap_to_half_int(W_sol)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    # Verify exactness with snapped values
                    # Re-compute residuals with snapped values
                    res_snap = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res_snap[i, j, k] = np.sum(U_snap[i * m + j] * V_snap[j * p + k] * W_snap[k * n + i])
                    
                    if np.allclose(res_snap, 1.0, atol=1e-9):
                        # Convert to lists
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
        except Exception as e:
            pass
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
