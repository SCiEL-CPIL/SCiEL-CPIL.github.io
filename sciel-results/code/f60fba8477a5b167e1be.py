import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 110.0
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W

    # Target: Find rank < 45
    # Known minimal rank is 20.
    
    def try_find_decomposition(rank, max_iters=100, restarts=5):
        # T is 1 for all i,j,k
        # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        
        for restart in range(restarts):
            # Initialize with small random integers
            U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
            
            # Normalize to avoid divergence
            # We can fix the scale of W to 1 for each column?
            # Or just use ALS.
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each i,j, U[i,j,:] = argmin sum_k (1 - V[j,k,:] * W[k,i,:]) * U[i,j,:]
                # This is a linear least squares problem for each (i,j)
                # Let y_{i,j} be the vector of residuals for fixed U? No.
                # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
                # For fixed V, W, U_{i,j,c} is determined by:
                # sum_k (1 - sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}) * V_{j,k,c} W_{k,i,c} = 0
                # This is non-linear in U because of the product V*W? No, V and W are fixed.
                # Let M_{i,j,k} = sum_c V_{j,k,c} W_{k,i,c}. This is a scalar for each k?
                # No, V and W are vectors.
                # Let's reshape.
                # For fixed i,j, we want to find u = U[i,j,:].
                # Minimize sum_k (1 - u^T (V[j,k,:] * W[k,i,:]))^2
                # Let A_k = V[j,k,:] * W[k,i,:] (element-wise product, length rank)
                # Minimize sum_k (1 - u^T A_k)^2
                # This is linear least squares: A u = 1
                # Where A is a matrix of size p x rank.
                # Row k of A is A_k.
                
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                # Solve A_mat u = ones(p)
                # Least squares
                u, residuals, rank_u, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                U[i*m+j, :] = u
                
                # Update V
                # For fixed U, W, find v = V[j,k,:]
                # Minimize sum_i (1 - sum_c U[i,j,c] v_c W[k,i,c])^2
                # Let B_i = U[i,j,:] * W[k,i,:]
                # Minimize sum_i (1 - v^T B_i)^2
                # B is n x rank.
                B_mat = np.zeros((n, rank))
                for i in range(n):
                    B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, residuals, rank_v, sv = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                V[j*p+k, :] = v
                
                # Update W
                # For fixed U, V, find w = W[k,i,:]
                # Minimize sum_j (1 - sum_c U[i,j,c] V[j,k,c] w_c)^2
                # Let C_j = U[i,j,:] * V[j,k,:]
                # Minimize sum_j (1 - w^T C_j)^2
                # C is m x rank.
                C_mat = np.zeros((m, rank))
                for j in range(m):
                    C_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, residuals, rank_w, sv = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)
                W[k*n+i, :] = w
                
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        if not np.isclose(s, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_r, V_r, W_r
        
        return None

    # Try ranks 20 to 30
    for r in range(20, 31):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = get_schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
