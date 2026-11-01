"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 52
    # Try to find a lower rank decomposition
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the tensor
    # T[i,j,k] = 1 if this term appears in C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has shape (n*p, m*p, n*m) in the W[k*n+i, j*p+k, i*m+j] convention
    # Actually: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] should reconstruct T
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    def build_tensor():
        """Return the 3D tensor as a numpy array of shape (n*m, m*p, p*n)"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    T = build_tensor()
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
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
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try rank minimization using iterative methods
    # We'll try to find a decomposition of rank r < 52
    
    # Use a simple approach: try various ranks and use alternating least squares
    # to minimize the reconstruction error, then check if we can round to integers/half-integers
    
    for target_rank in range(40, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        np.random.seed(42 + target_rank)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Alternating minimization
        T_flat = T.reshape(n*m * m*p, p*n)
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each column r, minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]||
            # Fix V and W, update U
            # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed j,k: T[:,j,k] = U[:,r] * (V[j,r] * W[k,r])
            
            # More efficient: 
            # For each r, U[:,r] should satisfy: sum_r U[:,r] (V[:,r] outer W[:,r]) = T
            # This is a linear system in U[:,r]
            
            # Build the linear system for U
            # T.reshape(n*m, -1) = U @ (V * W^T).T  -- not quite
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Let M[j,k] = sum_r V[j,r] W[k,r]  -- no, that's not right either
            
            # Actually: for fixed r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # We need sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
            
            # To update U: fix V, W. Then T = sum_r U[:,r] ⊗ (V[:,r] ⊗ W[:,r])
            # Reshape: T.reshape(n*m, m*p*p*n) = U @ (V ⊗ W).reshape(target_rank, m*p*p*n)
            
            VW = (V[:, :, None] * W[None, :, :]).reshape(target_rank, m*p * p*n)
            # T.reshape(n*m, m*p*p*n) = U @ VW
            T_reshaped = T.reshape(n*m, m*p * p*n)
            
            # Solve least squares: U = T_reshaped @ VW^+ 
            try:
                U_new = np.linalg.lstsq(VW, T_reshaped, rcond=None)[0].T
            except:
                break
            U = U_new
            
            # Update V similarly
            UW = (U[:, :, None] * W[None, :, :]).reshape(target_rank, n*m * p*n)
            T_reshaped_v = T.reshape(m*p, n*m * p*n)
            try:
                V_new = np.linalg.lstsq(UW, T_reshaped_v, rcond=None)[0].T
            except:
                break
            V = V_new
            
            # Update W similarly
            UV = (U[:, :, None] * V[None, :, :]).reshape(target_rank, n*m * m*p)
            T_reshaped_w = T.reshape(p*n, n*m * m*p)
            try:
                W_new = np.linalg.lstsq(UV, T_reshaped_w, rcond=None)[0].T
            except:
                break
            W = W_new
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(recon - T)
            if error < 1e-6:
                break
        
        # Check if we got a good solution
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(recon - T)
        
        if error < 1e-4:
            # Try to round to integers or half-integers
            # Scale to make entries close to integers/half-integers
            all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            nonzero = all_entries[np.abs(all_entries) > 1e-8]
            if len(nonzero) > 0:
                # Try to find a scaling that makes entries close to half-integers
                # For now, just check if entries are already close to half-integers
                max_dev = 0
                for entry in nonzero:
                    nearest_half = np.round(entry * 2) / 2
                    max_dev = max(max_dev, abs(entry - nearest_half))
                
                if max_dev < 1e-3:
                    # Round to half-integers
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    error = np.linalg.norm(recon - T)
                    if error < 1e-6:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
