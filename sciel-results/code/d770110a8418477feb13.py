import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Trivial rank-18 decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Try to reduce rank by finding dependencies
    # Strategy: attempt rank r = 17, 16, ... using SVD-based approximation
    for target_rank in range(trivial_rank - 1, 14, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Build the tensor T[i,j,k] = delta_{i,j,k} for the multiplication tensor
        # T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 if this term contributes
        # Actually, the tensor for matrix mult (A 2x3) * (B 3x3) = C 2x3
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # But we need to be more careful about the structure
        
        # The multiplication tensor M has entries M[i, j, k] = 1 
        # where i indexes rows of A (and C), j indexes cols of A / rows of B, k indexes cols of B (and C)
        # M[i,j,k] = 1 for all valid i,j,k in this case (it's the full tensor)
        
        # Reshape to matrix form for SVD
        # U matrix: rows = (n*m) = 6, cols = rank
        # V matrix: rows = (m*p) = 9, cols = rank  
        # W matrix: rows = (p*n) = 6, cols = rank
        
        # The constraint is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # where a = i*m+j (6 values), b = j*p+k (9 values), c = k*n+i (6 values)
        
        # For a given rank r, we can try random initialization and gradient descent
        # Or use alternating least squares
        
        # Try ALS (Alternating Least Squares)
        np.random.seed(42)
        U_curr = np.random.randn(n * m, target_rank) * 0.1
        V_curr = np.random.randn(m * p, target_rank) * 0.1
        W_curr = np.random.randn(p * n, target_rank) * 0.1
        
        # Initialize with partial trivial solution
        for r in range(min(target_rank, trivial_rank)):
            if r < n * m:
                U_curr[r, r % target_rank] = 1
            if r < m * p:
                V_curr[r, r % target_rank] = 1
            if r < p * n:
                W_curr[r, r % target_rank] = 1
        
        # ALS iterations
        max_als_iter = 50
        tol = 1e-8
        
        for it in range(max_als_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Update U: for each row a of U, solve least squares
            # sum_r U[a,r] * (V * W)_r = T[a, :, :] reshaped
            # (V * W)_r = V[:, r] outer W[:, r] reshaped to (m*p, p*n) -> need to match indices
            
            # Actually, let's think about this differently
            # The tensor equation: for each (a, b, c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # where a = i*m+j, b = j*p+k, c = k*n+i
            
            # For fixed V and W, update U:
            # For each a: sum_r U[a,r] * sum_b' V[b',r] W[c',r] ... this is complex
            
            # Let me use a different approach: flatten and use SVD
            # Build the "unfolded" matrices
            
            # Unfold along first mode: T_(1) of shape (n*m, m*p*p*n)
            # T_(1)[a, (b,c)] = T[a,b,c]
            # Then U * (V ⊗ W)^T = T_(1)
            # So we want rank-r approximation of T_(1)
            
            # Build T_(1)
            T_1 = np.zeros((n * m, m * p * p * n))
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        T_1[a, b * (p * n) + c] = 1.0
            
            # SVD of T_1
            U_svd, s, Vt_svd = np.linalg.svd(T_1, full_matrices=False)
            
            # Keep top target_rank components
            U_new = U_svd[:, :target_rank] * s[:target_rank]
            VW_product = Vt_svd[:target_rank, :].T  # shape (m*p*p*n, target_rank)
            
            # Now factor VW_product = V ⊗ W
            # VW_product[b * (p*n) + c, r] = V[b, r] * W[c, r]
            # For each r, VW_product[:, r] = V[:, r] outer W[:, r]
            # So V[:, r] and W[:, r] are rank-1 factors of VW_product[:, r]
            
            V_new = np.zeros((m * p, target_rank))
            W_new = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                vec = VW_product[:, r]  # shape (m*p*p*n,)
                mat = vec.reshape(m * p, p * n)
                # Rank-1 approximation of mat
                if np.linalg.norm(mat) > 1e-10:
                    u_mat, s_mat, vt_mat = np.linalg.svd(mat, full_matrices=False)
                    if len(s_mat) > 0 and s_mat[0] > 1e-10:
                        V_new[:, r] = u_mat[:, 0] * np.sqrt(s_mat[0])
                        W_new[:, r] = vt_mat[0, :] * np.sqrt(s_mat[0])
                    else:
                        V_new[:, r] = 0
                        W_new[:, r] = 0
                else:
                    V_new[:, r] = 0
                    W_new[:, r] = 0
            
            # Check convergence
            U_curr, V_curr, W_curr = U_new, V_new, W_new
        
        # Verify the decomposition
        error = 0.0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    expected = 1.0
                    error += (val - expected) ** 2
        
        if error < 1e-6:
            # Found a valid decomposition with lower rank!
            best_U = U_curr
            best_V = V_curr
            best_W = W_curr
            best_rank = target_rank
            if best_rank <= 16:
                break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest half-integer if close
    def round_to_half(x):
        rounded = np.round(2 * x) / 2
        if np.allclose(x, rounded, atol=1e-6):
            return rounded
        return x
    
    U_list = round_to_half(np.array(U_list)).tolist()
    V_list = round_to_half(np.array(V_list)).tolist()
    W_list = round_to_half(np.array(W_list)).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
