import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Trivial rank
    R_max = n * m * p
    
    # Precompute index mappings for the trivial solution
    # We map r in 0..R_max-1 to (i, j, k)
    # r = i * m * p + j * p + k
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Generate trivial solution matrices
    U_triv = np.zeros((n_m, R_max))
    V_triv = np.zeros((m_p, R_max))
    W_triv = np.zeros((p_n, R_max))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_triv[u_idx, r] = 1.0
                V_triv[v_idx, r] = 1.0
                W_triv[w_idx, r] = 1.0

    # Try ranks from R_max - 1 downwards
    # We limit the search to save time
    max_search_depth = 20
    
    for rank_offset in range(1, max_search_depth + 1):
        R = R_max - rank_offset
        if R < 20: # Don't go too low, lower bound is 20
            break
            
        # Initialize with first R columns of trivial solution
        # This drops the last 'rank_offset' terms
        U = U_triv[:, :R].copy()
        V = V_triv[:, :R].copy()
        W = W_triv[:, :R].copy()
        
        # Add small random noise to avoid symmetry issues
        np.random.seed(42 + rank_offset)
        U += np.random.randn(n_m, R) * 1e-3
        V += np.random.randn(m_p, R) * 1e-3
        W += np.random.randn(p_n, R) * 1e-3
        
        # ALS Optimization
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(100):
            # Update U
            # U[i*m+j, :] needs to satisfy sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed V, W, the term for U[i*m+j, r] is linear.
            # Let A_u[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # We want U[i*m+j, :] @ A_u[i*m+j, :] = 1
            # Solution: U[i*m+j, :] = A_u[i*m+j, :] / (A_u[i*m+j, :] @ A_u[i*m+j, :])
            
            A_u = np.zeros((n_m, R))
            for i in range(n):
                for j in range(m):
                    # V slice: j*p to j*p+p
                    V_slice = V[j*p:(j+1)*p, :]
                    # W slice: indices k*n+i for k in 0..p-1
                    W_idx = np.array([k*n + i for k in range(p)])
                    W_slice = W[W_idx, :]
                    A_u[i*m+j, :] = np.sum(V_slice * W_slice, axis=0)
            
            norms_sq = np.sum(A_u * A_u, axis=1)
            norms_sq[norms_sq == 0] = 1e-9
            U = A_u / norms_sq[:, np.newaxis]
            
            # Update V
            # V[j*p+k, :] needs to satisfy sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1
            # A_v[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            
            A_v = np.zeros((m_p, R))
            for j in range(m):
                for k in range(p):
                    # U slice: indices i*m+j for i in 0..n-1
                    U_idx = np.array([i*m + j for i in range(n)])
                    U_slice = U[U_idx, :]
                    # W slice: indices k*n+i for i in 0..n-1
                    W_idx = np.array([k*n + i for i in range(n)])
                    W_slice = W[W_idx, :]
                    A_v[j*p+k, :] = np.sum(U_slice * W_slice, axis=0)
            
            norms_sq_v = np.sum(A_v * A_v, axis=1)
            norms_sq_v[norms_sq_v == 0] = 1e-9
            V = A_v / norms_sq_v[:, np.newaxis]
            
            # Update W
            # W[k*n+i, :] needs to satisfy sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # A_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            
            A_w = np.zeros((p_n, R))
            for k in range(p):
                for i in range(n):
                    # U slice: indices i*m+j for j in 0..m-1
                    U_idx = np.array([i*m + j for j in range(m)])
                    U_slice = U[U_idx, :]
                    # V slice: indices j*p+k for j in 0..m-1
                    V_idx = np.array([j*p + k for j in range(m)])
                    V_slice = V[V_idx, :]
                    A_w[k*n+i, :] = np.sum(U_slice * V_slice, axis=0)
            
            norms_sq_w = np.sum(A_w * A_w, axis=1)
            norms_sq_w[norms_sq_w == 0] = 1e-9
            W = A_w / norms_sq_w[:, np.newaxis]
            
            # Calculate Error
            err = 0.0
            # Vectorized error calculation
            # err = sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
            # We can compute this more efficiently
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
                        
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-8:
                break
                
        if best_error < 1e-6:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
                    
            if is_exact:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    # Fallback to trivial solution
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
# EVOLVE-BLOCK-END
