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
    
    # Try to find rank-70 solution using vectorized alternating minimization
    if time.time() - start_time < time_limit - 20:
        res_70 = try_find_rank_70_vectorized(n, m, p, start_time, time_limit)
        if res_70 is not None:
            return res_70
    
    return {"U": U, "V": V, "W": W}

def try_find_rank_70_vectorized(n, m, p, start_time, time_limit):
    """
    Try to find a rank-70 solution using vectorized alternating minimization.
    """
    rank = 70
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Precompute indices for vectorized operations
    # For U update: for each (i,j), we need V[j, :, :] and W[:, i, :]
    # V is (m, p, rank) if reshaped. W is (p, n, rank) if reshaped.
    
    # Reshape U, V, W to 3D for easier indexing?
    # U[i, j, c], V[j, k, c], W[k, i, c]
    
    for trial in range(50):
        if time.time() - start_time > time_limit - 5:
            break
        
        np.random.seed(42 + trial)
        
        # Initialization
        U = np.random.randn(n, m, rank) * 0.5
        V = np.random.randn(m, p, rank) * 0.5
        W = np.random.randn(p, n, rank) * 0.5
        
        # Flatten for storage? No, keep 3D for logic, flatten for solving if needed.
        
        for iteration in range(200):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Update U
            # For each i, j:
            # A[k, c] = V[j, k, c] * W[k, i, c]
            # b[k] = 1
            # Solve A^T A u = A^T b
            
            # Construct A^T A for all i, j
            # A is (p, rank) for each (i, j).
            # A^T A is (rank, rank).
            # A^T A[c1, c2] = sum_k V[j, k, c1] * W[k, i, c1] * V[j, k, c2] * W[k, i, c2]
            # This can be computed using einsum.
            
            # V[j, k, c] * W[k, i, c] -> Element-wise product M[j, k, i, c]
            # M = V[:, :, None, :] * W[None, :, :, :] # Shape (m, p, n, rank)
            # M[j, k, i, c] = V[j, k, c] * W[k, i, c]
            
            M = V[:, :, np.newaxis, :] * W[np.newaxis, :, :, :] # (m, p, n, rank)
            
            # A^T A for (i, j) is sum_k M[j, k, i, :] * M[j, k, i, :].T
            # Let's compute N_U[i, j, c1, c2] = sum_k M[j, k, i, c1] * M[j, k, i, c2]
            # N_U = np.einsum('jkic, jkid -> ijcd', M, M) # Shape (n, m, rank, rank)
            # Wait, indices: M is (m, p, n, rank).
            # We want output (n, m, rank, rank).
            # Sum over k (which is index 1 in M).
            # M[j, k, i, c1] * M[j, k, i, c2]
            # einsum: 'jkic, jkid -> ijcd'
            
            N_U = np.einsum('jkic, jkid -> ijcd', M, M)
            
            # A^T b for (i, j) is sum_k M[j, k, i, c] * 1 = sum_k M[j, k, i, c]
            # b_U[i, j, c] = sum_k M[j, k, i, c]
            b_U = np.einsum('jkic -> ijc', M) # Shape (n, m, rank)
            
            # Solve N_U[i, j] u[i, j] = b_U[i, j]
            # N_U shape (n, m, rank, rank). b_U shape (n, m, rank).
            # np.linalg.solve supports batched solving.
            try:
                U_new = np.linalg.solve(N_U, b_U)
                U = U_new
            except np.linalg.LinAlgError:
                # Use lstsq if singular
                U = np.linalg.lstsq(N_U, b_U, rcond=None)[0]
            
            # Update V
            # For each j, k:
            # A[i, c] = U[i, j, c] * W[k, i, c]
            # b[i] = 1
            # A^T A[c1, c2] = sum_i U[i, j, c1] * W[k, i, c1] * U[i, j, c2] * W[k, i, c2]
            
            # M_V[i, j, k, c] = U[i, j, c] * W[k, i, c]
            # U shape (n, m, rank). W shape (p, n, rank).
            # We want M_V[i, j, k, c] = U[i, j, c] * W[k, i, c]
            # M_V = U[:, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :] # (n, m, p, rank)
            
            M_V = U[:, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :] # (n, m, p, rank)
            
            # N_V[j, k, c1, c2] = sum_i M_V[i, j, k, c1] * M_V[i, j, k, c2]
            N_V = np.einsum('ijkc, ijkd -> jkcd', M_V, M_V) # Shape (m, p, rank, rank)
            
            # b_V[j, k, c] = sum_i M_V[i, j, k, c]
            b_V = np.einsum('ijkc -> jkc', M_V) # Shape (m, p, rank)
            
            try:
                V_new = np.linalg.solve(N_V, b_V)
                V = V_new
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(N_V, b_V, rcond=None)[0]
            
            # Update W
            # For each k, i:
            # A[j, c] = U[i, j, c] * V[j, k, c]
            # b[j] = 1
            # A^T A[c1, c2] = sum_j U[i, j, c1] * V[j, k, c1] * U[i, j, c2] * V[j, k, c2]
            
            # M_W[i, j, k, c] = U[i, j, c] * V[j, k, c]
            # U shape (n, m, rank). V shape (m, p, rank).
            # M_W = U[:, :, np.newaxis, :] * V[np.newaxis, :, :, :] # (n, m, p, rank)
            
            M_W = U[:, :, np.newaxis, :] * V[np.newaxis, :, :, :] # (n, m, p, rank)
            
            # N_W[k, i, c1, c2] = sum_j M_W[i, j, k, c1] * M_W[i, j, k, c2]
            N_W = np.einsum('ijkc, ijkd -> kicd', M_W, M_W) # Shape (p, n, rank, rank)
            
            # b_W[k, i, c] = sum_j M_W[i, j, k, c]
            b_W = np.einsum('ijkc -> kic', M_W) # Shape (p, n, rank)
            
            try:
                W_new = np.linalg.solve(N_W, b_W)
                W = W_new
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(N_W, b_W, rcond=None)[0]
            
            # Check error
            # T_est[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # T_est = np.einsum('ijc, jkc, kic -> ijk', U, V, W)
            T_est = np.einsum('ijc, jkc, kic -> ijk', U, V, W)
            error = np.sum((T_est - 1.0) ** 2)
            
            if error < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_est_r = np.einsum('ijc, jkc, kic -> ijk', U_r, V_r, W_r)
                error_r = np.sum((T_est_r - 1.0) ** 2)
                
                if error_r < 1e-6:
                    return {
                        "U": U_r.reshape(n*m, rank).tolist(),
                        "V": V_r.reshape(m*p, rank).tolist(),
                        "W": W_r.reshape(p*n, rank).tolist()
                    }
        
    return None
