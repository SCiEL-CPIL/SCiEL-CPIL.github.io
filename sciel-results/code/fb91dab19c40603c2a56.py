"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 if i==i'', j==j', k==k''
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Schoolbook solution as initial best
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(4, 12, 12) = 12
    # Also, rank >= max(n, m, p) = 6, but tighter bounds exist
    # Try ranks from 20 down to 14
    
    for target_rank in range(20, 13, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random initializations
        n_trials = 5
        for trial in range(n_trials):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating least squares
            converged = False
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each c, U[:,c] should satisfy
                # T[:, :, :] ≈ U @ V.T @ W.T (in some sense)
                # Actually, we need: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] ≈ T
                # Reshape T to (n*m, m*p*p*n) and solve
                
                # For U: fix V and W, solve for U
                # T_reshaped[a, b, c_idx] = sum_r U[a, r] * V[b, r] * W[c_idx, r]
                # This is a linear system in U
                # T_reshaped[a, b, c_idx] = sum_r U[a, r] * (V[b, r] * W[c_idx, r])
                # Let M_U[b, c_idx, r] = V[b, r] * W[c_idx, r]
                # Then T_reshaped[a, :, :] = U[a, :] @ M_U[:, :, :].T ... 
                
                # Better: reshape T to (n*m, m*p, p*n)
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, c: T[:, b, c] = U @ (V[b, :] * W[c, :])^T ... no
                # T[:, b, c] = sum_r U[:, r] * V[b, r] * W[c, r]
                # = U @ diag(V[b, :] * W[c, :]) ... not quite
                # Let v_w = V[b, :] * W[c, :] (element-wise product), shape (rank,)
                # T[:, b, c] = U @ v_w
                # So for each (b,c), we have U @ v_w = T[:, b, c]
                # Stack all (b,c) pairs: 
                # Let A_mat[b,c,:] = V[b,:] * W[c,:]  (shape m*p, p*n, rank)
                # Then T_2d[a, b*p*n + c] = sum_r U[a,r] * A_mat[b,c,r]
                # T_2d = U @ A_mat_reshaped.T where A_mat_reshaped has shape (m*p*p*n, rank)
                
                A_mat = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, rank)
                A_mat_flat = A_mat.reshape(m*p*p*n, target_rank)
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Solve U @ A_mat_flat.T = T_flat
                # U = T_flat @ A_mat_flat.T @ (A_mat_flat @ A_mat_flat.T)^{-1}
                try:
                    ATA = A_mat_flat @ A_mat_flat.T  # (n*m, n*m)
                    ATb = A_mat_flat @ T_flat.T  # (n*m, m*p*p*n)
                    # U[a, r] = sum_b T_flat[a,b] * A_mat_flat[b,r] / (ATA inverse)
                    # Actually: U = T_flat @ A_mat_flat.T @ inv(A_mat_flat @ A_mat_flat.T)
                    # Wait, dimensions: T_flat is (n*m, m*p*p*n), A_mat_flat is (m*p*p*n, rank)
                    # U should be (n*m, rank)
                    # T_flat = U @ A_mat_flat.T  =>  T_flat @ A_mat_flat = U @ (A_mat_flat.T @ A_mat_flat)
                    # So U = T_flat @ A_mat_flat @ inv(A_mat_flat.T @ A_mat_flat)
                    A_TA = A_mat_flat.T @ A_mat_flat  # (rank, rank)
                    T_A = T_flat @ A_mat_flat  # (n*m, rank)
                    U_new = np.linalg.solve(A_TA, T_A.T).T
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                A_mat = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, rank)
                A_mat_flat = A_mat.reshape(n*m*p*n, target_rank)
                T_flat = T.reshape(m*p, n*m*p*n)
                try:
                    A_TA = A_mat_flat.T @ A_mat_flat
                    T_A = T_flat @ A_mat_flat
                    V_new = np.linalg.solve(A_TA, T_A.T).T
                except np.linalg.LinAlgError:
                    break
                
                # Update W
                A_mat = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, rank)
                A_mat_flat = A_mat.reshape(n*m*m*p, target_rank)
                T_flat = T.reshape(p*n, n*m*m*p)
                try:
                    A_TA = A_mat_flat.T @ A_mat_flat
                    T_A = T_flat @ A_mat_flat
                    W_new = np.linalg.solve(A_TA, T_A.T).T
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                U, V, W = U_new, V_new, W_new
                # Compute error
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Try to round to exact values
                # Check if entries are close to integers or half-integers
                all_close = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) > 1e-6 and abs(val - nearest_half) > 1e-6:
                            all_close = False
                            break
                    if not all_close:
                        break
                
                if all_close:
                    # Round to exact
                    U_exact = np.round(U * 2) / 2
                    V_exact = np.round(V * 2) / 2
                    W_exact = np.round(W * 2) / 2
                    
                    # Verify exactness
                    T_recon_exact = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_recon_exact, T, atol=1e-10):
                        # Convert to lists
                        U_list = U_exact.tolist()
                        V_list = V_exact.tolist()
                        W_list = W_exact.tolist()
                        # Convert to int where possible
                        def to_exact_list(lst):
                            result = []
                            for row in lst:
                                new_row = []
                                for val in row:
                                    if abs(val - round(val)) < 1e-10:
                                        new_row.append(int(round(val)))
                                    elif abs(val * 2 - round(val * 2)) < 1e-10:
                                        new_row.append(int(round(val * 2)) / 2)
                                    else:
                                        new_row.append(val)
                                result.append(new_row)
                            return result
                        U_list = to_exact_list(U_list)
                        V_list = to_exact_list(V_list)
                        W_list = to_exact_list(W_list)
                        
                        best_result = (U_list, V_list, W_list)
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
        if time.time() > deadline - 10:
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

