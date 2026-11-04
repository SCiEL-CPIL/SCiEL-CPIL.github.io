import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_schoolbook():
    n, m, p = 4, 5, 5
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * (n * m)
                u[i * m + j] = 1.0
                v = [0.0] * (m * p)
                v[j * p + k] = 1.0
                w = [0.0] * (p * n)
                w[k * n + i] = 1.0
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit

    # Build schoolbook decomposition (rank 100)
    U_school, V_school, W_school = _build_schoolbook()

    # Convert to numpy for efficient operations
    U_s = np.array(U_school, dtype=np.float64)
    V_s = np.array(V_school, dtype=np.float64)
    W_s = np.array(W_school, dtype=np.float64)

    # Target tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i * m + j, j * p + k, k * n + i] = 1.0

    best_rank = n * m * p  # 100
    best_U = U_s.copy()
    best_V = V_s.copy()
    best_W = W_s.copy()

    # Try to find lower rank using ALS optimization
    # We'll try ranks from 99 down to a reasonable minimum
    # For a 4x5x5 matrix multiplication, the known lower bound is around 68-70
    # but finding exact decompositions is hard. Let's try a few.
    
    np.random.seed(42)
    
    ranks_to_try = list(range(99, 60, -1))  # Try ranks 99 down to 61
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple restarts for this rank
        max_attempts = 2
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with warm start from schoolbook + noise
            U = U_s[:, :rank].copy()
            V = V_s[:, :rank].copy()
            W = W_s[:, :rank].copy()
            
            # Add small random perturbation to break symmetry
            noise_scale = 0.01
            U += np.random.randn(n * m, rank) * noise_scale
            V += np.random.randn(m * p, rank) * noise_scale
            W += np.random.randn(p * n, rank) * noise_scale
            
            # ALS iterations
            max_iter = 50
            prev_residual = np.inf
            
            for iter_num in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: For each column c, U[:, c] = argmin sum_a ||sum_b T[a,b,:] V[b,c] W[:,c] - target||
                # More precisely: U_new[a, c] = sum_{b,k} T[a,b,k] V[b,c] W[k,c] / (sum_{b,k} T[a,b,k] W[k,c]^2 + eps)
                # But for tensor decomposition T = sum_r U[:,r] V[:,r] W[:,r],
                # the normal equations give:
                # U[:,r] = (T contracted with V[:,r] and W[:,r])
                
                # Contract T with V and W for each r
                # T[a, b, k] V[b, r] W[k, r] summed over b, k
                U_new = np.einsum('abk,br,kr->ar', target_tensor, V, W)
                
                # Normalization: denominator = sum_{b,k} T[a,b,k] W[k,r]^2 ... 
                # Actually for the standard ALS:
                # U[:,r] = T_{a, V[:,r], W[:,r]} / (V[:,r]^T T_{a,.,.} W[:,r])
                # Let's compute denominator
                denom_U = np.einsum('abk,br,kr,br,kr->ar', target_tensor, V, W, V, W)
                denom_U = np.maximum(denom_U, 1e-12)
                U_new = U_new / denom_U
                
                # Update V
                V_new = np.einsum('abk,ar,kr->br', target_tensor, U, W)
                denom_V = np.einsum('abk,ar,kr,ar,kr->br', target_tensor, U, W, U, W)
                denom_V = np.maximum(denom_V, 1e-12)
                V_new = V_new / denom_V
                
                # Update W
                W_new = np.einsum('abk,ar,br->kr', target_tensor, U, V)
                denom_W = np.einsum('abk,ar,br,ar,br->kr', target_tensor, U, V, U, V)
                denom_W = np.maximum(denom_W, 1e-12)
                W_new = W_new / denom_W
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if iter_num > 0:
                    # Quick residual check every 10 iterations
                    if iter_num % 10 == 0:
                        T_approx = np.einsum('ar,br,kr->abk', U, V, W)
                        residual = np.linalg.norm(target_tensor - T_approx)
                        if residual < 1e-8:
                            found = True
                            break
                        if abs(residual - prev_residual) < 1e-10:
                            prev_residual = residual
                        else:
                            prev_residual = residual
            
            if found:
                # Try to round to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_rounded = np.einsum('ar,br,kr->abk', U_rounded, V_rounded, W_rounded)
                residual_rounded = np.linalg.norm(target_tensor - T_approx_rounded)
                
                if residual_rounded < 1e-9:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
                else:
                    # Keep the unrounded version if it's close enough
                    T_approx_unrounded = np.einsum('ar,br,kr->abk', U, V, W)
                    residual_unrounded = np.linalg.norm(target_tensor - T_approx_unrounded)
                    if residual_unrounded < 1e-6:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break
        
        if found:
            # If we found a valid decomposition at this rank, we can try lower ranks
            # But for safety, let's continue to see if we can go lower
            pass
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
