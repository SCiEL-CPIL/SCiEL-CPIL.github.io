"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor
    # Actually, the matmul tensor is: T[i,j,k] = delta (it's 1 for the "identity" 
    # structure). Let me reconsider.
    
    # The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the 
    # appropriate subspace). Wait, no.
    
    # Actually the matmul tensor T is defined by T[i,j,k] = 1 if we're computing 
    # C[i,k] from A[i,j]*B[j,k]. The tensor has T[i,j,k] = 1 for all valid indices,
    # meaning it's the all-ones tensor of shape (n, m, p).
    
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's use ALS to find a low-rank decomposition.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Build the target tensor
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the tensor."""
        # Compute T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        rank = U.shape[1]
        # Reshape for efficient computation
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # This is a batched operation
        # For each (i,j,k): dot(U_reshaped[i,j,:], V_reshaped[j,k,:] * W_reshaped[k,i,:])
        
        # Use einsum: sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        max_err = np.max(np.abs(T_recon - T))
        return max_err < tol
    
    def als_decompose(rank, max_iter=200, tol=1e-10, seed=None):
        """ALS decomposition for the matmul tensor."""
        rng = np.random.default_rng(seed)
        
        # Initialize
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        prev_err = np.inf
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), U[i,j,:] should satisfy
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for all k
            # So for fixed (i,j): T[i,j,:] = U[i,j,:] @ (V[j,:,:] * W[:,i,:])^T ... 
            # Actually: for fixed (i,j), and for each k:
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # Let M[i,j,k,c] = V[j,k,c] * W[k,i,c]
            # Then T[i,j,:] = U[i,j,:] @ M[i,j,:,:].T
            # So U[i,j,:] = solve(M[i,j,:,:].T @ M[i,j,:,:], M[i,j,:,:].T @ T[i,j,:])
            
            for i in range(n):
                for j in range(m):
                    # M[k,c] = V[j,k,c] * W[k,i,c]
                    M = V_reshaped[j, :, :] * W_reshaped[:, i, :]  # shape (p, rank)
                    # Solve: M.T @ U[i,j,:] = T[i,j,:]
                    A_mat = M.T @ M  # (rank, rank)
                    b_vec = M.T @ T[i, j, :]  # (rank,)
                    try:
                        U_reshaped[i, j, :] = np.linalg.solve(A_mat + 1e-12 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        U_reshaped[i, j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update V: for each (j,k), 
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (j,k): T[:,j,k] = V[j,k,:] @ (U[:,j,:] * W[k,:])^T ... 
            # M[i,c] = U[i,j,c] * W[k,i,c]
            # T[:,j,k] = V[j,k,:] @ M.T
            # V[j,k,:] = solve(M @ M.T, T[:,j,k] @ M)
            
            for j in range(m):
                for k in range(p):
                    M = U_reshaped[:, j, :] * W_reshaped[k, :, :]  # shape (n, rank)
                    A_mat = M @ M.T  # (n, n)
                    b_vec = T[:, j, k] @ M  # (rank,)
                    try:
                        V_reshaped[j, k, :] = np.linalg.solve(A_mat + 1e-12 * np.eye(n), b_vec)
                    except np.linalg.LinAlgError:
                        V_reshaped[j, k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update W: for each (k,i),
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (k,i): T[i,:,k] = W[k,i,:] @ (U[i,:,:] * V[:,k,:])^T ...
            # M[j,c] = U[i,j,c] * V[j,k,c]
            # T[i,:,k] = W[k,i,:] @ M.T
            # W[k,i,:] = solve(M @ M.T, T[i,:,k] @ M)
            
            for k in range(p):
                for i in range(n):
                    M = U_reshaped[i, :, :] * V_reshaped[:, k, :]  # shape (m, rank)
                    A_mat = M @ M.T  # (m, m)
                    b_vec = T[i, :, k] @ M  # (rank,)
                    try:
                        W_reshaped[k, i, :] = np.linalg.solve(A_mat + 1e-12 * np.eye(m), b_vec)
                    except np.linalg.LinAlgError:
                        W_reshaped[k, i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Compute error
            T_recon = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            err = np.max(np.abs(T_recon - T))
            
            if err < prev_err:
                prev_err = err
            if err < tol:
                break
            if err >= prev_err * 0.999:  # stalled
                break
        
        return U_reshaped.reshape(n*m, rank), V_reshaped.reshape(m*p, rank), W_reshaped.reshape(p*n, rank), prev_err
    
    # Try various ranks
    best_result = None
    best_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, _ = schoolbook()
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    min_rank = max(n * p, m * p, n * m)
    
    # Try several ranks with multiple restarts
    ranks_to_try = list(range(min_rank, best_rank + 1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break
        
        # Try a few restarts for each rank
        max_restarts = 5 if rank < 50 else 3 if rank < 70 else 2
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            seed = rank * 1000 + restart
            U, V, W, err = als_decompose(rank, max_iter=100, tol=1e-10, seed=seed)
            
            if err < 1e-6:
                # Found a valid decomposition
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W, rank)
                break  # Move to next rank
    
    U, V, W, rank = best_result
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The entries should be integer or half-integer
    # Let's check if they're close to such values
    def clean_matrix(M):
        """Round entries to nearest integer or half-integer if close."""
        M_clean = M.copy()
        for i in range(M_clean.shape[0]):
            for j in range(M_clean.shape[1]):
                val = M_clean[i, j]
                # Check if close to integer or half-integer
                int_val = round(val)
                half_int_val = round(2 * val) / 2
                if abs(val - int_val) < 1e-4:
                    M_clean[i, j] = int_val
                elif abs(val - half_int_val) < 1e-4:
                    M_clean[i, j] = half_int_val
        return M_clean
    
    U_clean = clean_matrix(U)
    V_clean = clean_matrix(V)
    W_clean = clean_matrix(W)
    
    # Verify the cleaned version
    if verify(U_clean, V_clean, W_clean, tol=1e-4):
        U, V, W = U_clean, V_clean, W_clean
    else:
        # Keep the original if cleaning broke it
        pass
    
    # Convert numpy arrays to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

