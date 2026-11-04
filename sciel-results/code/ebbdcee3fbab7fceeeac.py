import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    
    # Deadline guard: aim to finish well under 150s
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Build the sparse tensor representation
    # T[u, v, w] = 1 where u=i*m+j, v=j*p+k, w=k*n+i
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_indices.append((u, v, w))
    
    num_terms = len(tensor_indices)  # 60
    
    u_idx = np.array([t[0] for t in tensor_indices])
    v_idx = np.array([t[1] for t in tensor_indices])
    w_idx = np.array([t[2] for t in tensor_indices])
    
    # Precompute index arrays for efficient ALS
    # For updating U: need V[v_idx, :] and W[w_idx, :]
    # For updating V: need U[u_idx, :] and W[w_idx, :]
    # For updating W: need U[u_idx, :] and V[v_idx, :]
    
    def _check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition exactly reconstructs the tensor."""
        # For each non-zero entry (u, v, w), verify sum_r U[u,r]*V[v,r]*W[w,r] == 1
        # For all other entries, verify sum_r U[u,r]*V[v,r]*W[w,r] == 0
        # We only check the non-zero entries for speed, plus a few zero entries
        for (u, v, w) in tensor_indices:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val - 1.0) > tol:
                return False
        # Check some zero entries
        # u=0, v=0, w=1: i=0,j=0,k=0 gives w=0*2+0=0, not 1
        # Let's check a few random zero entries
        for _ in range(20):
            u = np.random.randint(0, dim_u)
            v = np.random.randint(0, dim_v)
            w = np.random.randint(0, dim_w)
            # Check if this is a tensor entry
            is_tensor = any(t[0]==u and t[1]==v and t[2]==w for t in tensor_indices)
            if not is_tensor:
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val) > tol:
                    return False
        return True
    
    def _try_rank_als(rank, max_iter=200, tol=1e-10, seed=0, time_budget=10.0):
        """Try to find a rank decomposition using ALS."""
        np.random.seed(seed)
        
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > start_time + time_budget:
                break
            
            # Update U: fix V, W
            # For each entry (u,v,w) in tensor: U[u,r] * V[v,r] * W[w,r] should sum to 1
            # A_U[idx, r] = V[v_idx[idx], r] * W[w_idx[idx], r]
            # b_U[idx] = 1
            # Solve A_U @ U = b_U (least squares)
            A_U = np.outer(V[v_idx, :].reshape(-1, rank), W[w_idx, :].reshape(-1, rank)).reshape(num_terms, rank)
            # Actually: A_U[idx, r] = V[v_idx[idx], r] * W[w_idx[idx], r]
            A_U = (V[v_idx, :] * W[w_idx, :])  # shape (num_terms, rank)
            b_U = np.ones(num_terms)
            
            # Solve least squares
            try:
                U_new, residuals, rank_au, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
            except:
                break
            
            # Update V: fix U, W
            A_V = (U[u_idx, :] * W[w_idx, :])  # shape (num_terms, rank)
            b_V = np.ones(num_terms)
            try:
                V_new, residuals, rank_av, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
            except:
                break
            
            # Update W: fix U, V
            A_W = (U[u_idx, :] * V[v_idx, :])  # shape (num_terms, rank)
            b_W = np.ones(num_terms)
            try:
                W_new, residuals, rank_aw, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Compute residual
            residual = 0.0
            for idx in range(num_terms):
                u, v, w = u_idx[idx], v_idx[idx], w_idx[idx]
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                residual += (val - 1.0) ** 2
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_residual < tol:
                break
        
        # Check validity
        if best_residual < 1e-8:
            # Try to rationalize
            U_r, V_r, W_r = _rationalize(best_U, best_V, best_W)
            if U_r is not None:
                if _check_validity(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r
        return None
    
    def _rationalize(U, V, W, tol=1e-6):
        """Try to convert float matrices to rational (half-integer) entries."""
        try:
            # Scale to find common denominator
            # Try small denominators
            for denom in [1, 2, 4, 8, 16, 32]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                # Check if close to original
                if (np.max(np.abs(U_r - U)) < tol and 
                    np.max(np.abs(V_r - V)) < tol and 
                    np.max(np.abs(W_r - W)) < tol):
                    # Convert to list of lists with half-integer entries
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    # Ensure entries are half-integers
                    valid = True
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for x in row:
                                if abs(x * 2 - round(x * 2)) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return U_list, V_list, W_list
            return None
        except:
            return None
    
    def _schoolbook():
        """Return the schoolbook rank-60 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(dim_u)]
        V = [[0.0] * rank for _ in range(dim_v)]
        W = [[0.0] * rank for _ in range(dim_w)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Strategy: Try to find a lower rank decomposition
    # Known lower bounds and results for matrix multiplication tensors:
    # For 2x2x2: rank 7 (Strassen)
    # For 2x3x3: rank 11? 
    # For 2x4x4: rank 20?
    # For 2x5x5: rank ?
    # For 2x5x6: rank ?
    
    # The schoolbook rank is 60. Let's try to find something better.
    # We'll try ranks from 30 up to 59, with a time budget for each.
    
    best_result = None
    best_rank = n * m * p  # 60
    
    # First, try some specific ranks that might be achievable
    # Use a hierarchical approach: decompose 2x5x6 into smaller parts
    
    # Actually, let's just try ALS with various ranks and seeds
    ranks_to_try = [30, 35, 40, 45, 50, 55, 59]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for each rank
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            
            result = _try_rank_als(rank, max_iter=100, tol=1e-10, seed=seed, time_budget=15.0)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = (U_r, V_r, W_r)
                best_rank = rank
                break  # Found a valid decomposition for this rank
        
        if best_result is not None:
            break  # Found a valid decomposition, move to lower rank
    
    # If no better decomposition found, return schoolbook
    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
