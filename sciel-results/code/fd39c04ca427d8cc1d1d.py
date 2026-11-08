import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Leave 5s margin for safety
    
    # Construct rank-71 solution using Strassen + schoolbook
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Try to find rank-70 solution using structured search
    if time.time() - start_time < time_limit - 20:
        res_70 = try_find_rank_70_structured(n, m, p, start_time, time_limit)
        if res_70 is not None:
            return res_70
    
    return {"U": U, "V": V, "W": W}

def try_find_rank_70_structured(n, m, p, start_time, time_limit):
    """
    Try to find a rank-70 solution by modifying the rank-71 solution.
    Strategy: The rank-71 solution has 4 Strassen blocks (28 rank) + 43 schoolbook terms.
    We need to eliminate one rank unit. 
    
    Approach: Use vectorized alternating minimization to find a rank-70 decomposition
    that matches the multiplication tensor.
    """
    rank = 70
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build the rank-71 solution as numpy arrays for initialization
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = len(blocks) * 7
    total_terms = n * m * p
    strassen_terms_count = len(blocks) * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank_71 = total_strassen_rank + schoolbook_terms_count
    
    U71 = np.zeros((n_m, total_rank_71))
    V71 = np.zeros((m_p, total_rank_71))
    W71 = np.zeros((p_n, total_rank_71))
    
    col = 0
    
    def add_strassen_block_np(i0, j0, k0, col_offset):
        c = col_offset + 0
        U71[i0*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 1
        U71[(i0+1)*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += -1
        
        c = col_offset + 2
        U71[i0*m + j0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += -1
        W71[(k0+1)*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 3
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[(j0+1)*p + k0, c] += 1
        V71[j0*p + k0, c] += -1
        W71[k0*n + i0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        
        c = col_offset + 4
        U71[i0*m + j0, c] += 1
        U71[i0*m + (j0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += -1
        W71[(k0+1)*n + i0, c] += 1
        
        c = col_offset + 5
        U71[(i0+1)*m + j0, c] += 1
        U71[i0*m + j0, c] += -1
        V71[j0*p + k0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 6
        U71[i0*m + (j0+1), c] += 1
        U71[(i0+1)*m + (j0+1), c] += -1
        V71[(j0+1)*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block_np(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U71[i*m+j, col] += 1
                    V71[j*p+k, col] += 1
                    W71[k*n+i, col] += 1
                    col += 1
    
    # Vectorized Alternating Minimization
    # Tensor T_ijk = 1 for all i,j,k
    # Constraint: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    for trial in range(30):
        if time.time() - start_time > time_limit - 5:
            break
        
        np.random.seed(42 + trial)
        
        # Initialization strategies
        if trial < 5:
            # Random small init
            U = np.random.randn(n_m, rank) * 0.5
            V = np.random.randn(m_p, rank) * 0.5
            W = np.random.randn(p_n, rank) * 0.5
        else:
            # Perturbed rank-71 (drop last col)
            U = U71[:, :rank].copy()
            V = V71[:, :rank].copy()
            W = W71[:, :rank].copy()
            # Add noise to escape local minima
            U += np.random.randn(n_m, rank) * 0.1
            V += np.random.randn(m_p, rank) * 0.1
            W += np.random.randn(p_n, rank) * 0.1
        
        # Pre-compute index arrays for vectorization
        # We need to update U, V, W iteratively.
        # Update U[i,j]: solve A U[i,j] = b where A[k,c] = V[j,k,c] * W[k,i,c]
        # Note: A depends on i and j. So we can't fully vectorize the solve across all i,j
        # without looping, but we can vectorize the matrix construction.
        
        for iteration in range(100):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Update U
            # For each i, j, we solve a linear system of size p x rank
            for i in range(n):
                for j in range(m):
                    # Construct A: (p, rank)
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # V_row = V[j*p:(j+1)*p, :] shape (p, rank)
                    # W_col = W[:, c] for each k? No, W[k*n+i, c]
                    # We need W indices [k*n+i for k in range(p)]
                    w_indices = np.array([k*n + i for k in range(p)])
                    W_sub = W[w_indices, :] # shape (p, rank)
                    V_sub = V[j*p:(j+1)*p, :] # shape (p, rank)
                    A = V_sub * W_sub # element-wise multiply, shape (p, rank)
                    b = np.ones(p)
                    
                    # Solve A x = b
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            # For each j, k, we solve A V[j,k] = b where A[i,c] = U[i*m+j, c] * W[k*n+i, c]
            for j in range(m):
                for k in range(p):
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    u_indices = np.array([i*m + j for i in range(n)])
                    U_sub = U[u_indices, :] # shape (n, rank)
                    w_indices = np.array([k*n + i for i in range(n)])
                    W_sub = W[w_indices, :] # shape (n, rank)
                    A = U_sub * W_sub # shape (n, rank)
                    b = np.ones(n)
                    
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W
            # For each k, i, we solve A W[k,i] = b where A[j,c] = U[i*m+j, c] * V[j*p+k, c]
            for k in range(p):
                for i in range(n):
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    u_indices = np.array([i*m + j for j in range(m)])
                    U_sub = U[u_indices, :] # shape (m, rank)
                    v_indices = np.array([j*p + k for j in range(m)])
                    V_sub = V[v_indices, :] # shape (m, rank)
                    A = U_sub * V_sub # shape (m, rank)
                    b = np.ones(m)
                    
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check error
            # Error = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            # Vectorize this check
            # Create tensors of shape (n, m, p, rank) ? Too big maybe.
            # n=3, m=5, p=5, rank=70. 3*5*5*70 = 5250 floats. Small.
            
            # Let's compute the tensor T_est
            # T_est[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # We can loop over i, j, k for the check, it's only 75 iterations.
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j, :]
                        v_val = V[j*p+k, :]
                        w_val = W[k*n+i, :]
                        dot = np.dot(u_val, v_val * w_val)
                        error += (dot - 1.0) ** 2
            
            if error < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded
                error_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dot = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            error_r += (dot - 1.0) ** 2
                
                if error_r < 1e-6:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
    return None
