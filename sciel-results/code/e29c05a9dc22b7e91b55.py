"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # safety margin before 150s wall-clock kill
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c] for all a,b,c."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=tol, rtol=0.0)
    
    # Schoolbook construction as fallback (rank 56)
    rank_school = n * m * p  # 56
    U_school = np.zeros((size_a, rank_school), dtype=np.float64)
    V_school = np.zeros((size_b, rank_school), dtype=np.float64)
    W_school = np.zeros((size_c, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    # Try the rank 2p = 28 construction
    # For each k, use 2 terms:
    # Term 0 (j=0): u[0]=1, u[2]=1, v[k]=1, w[2k]=1, w[2k+1]=1
    # Term 1 (j=1): u[1]=1, u[3]=1, v[14+k]=1, w[2k]=1, w[2k+1]=1
    rank_2p = 2 * p  # 28
    U_2p = np.zeros((size_a, rank_2p), dtype=np.float64)
    V_2p = np.zeros((size_b, rank_2p), dtype=np.float64)
    W_2p = np.zeros((size_c, rank_2p), dtype=np.float64)
    
    for k in range(p):
        # Term 0: covers j=0 contributions for both i values
        col0 = k
        U_2p[0][col0] = 1.0   # i=0, j=0
        U_2p[2][col0] = 1.0   # i=1, j=0
        V_2p[0 * p + k][col0] = 1.0  # j=0, k
        W_2p[k * n + 0][col0] = 1.0  # k, i=0
        W_2p[k * n + 1][col0] = 1.0  # k, i=1
        
        # Term 1: covers j=1 contributions for both i values
        col1 = p + k
        U_2p[1][col1] = 1.0   # i=0, j=1
        U_2p[3][col1] = 1.0   # i=1, j=1
        V_2p[1 * p + k][col1] = 1.0  # j=1, k
        W_2p[k * n + 0][col1] = 1.0  # k, i=0
        W_2p[k * n + 1][col1] = 1.0  # k, i=1
    
    # Verify the rank 28 construction
    if verify(U_2p, V_2p, W_2p):
        # Convert to integer lists for clean output
        return {"U": U_2p.astype(int).tolist(), "V": V_2p.astype(int).tolist(), "W": W_2p.astype(int).tolist()}
    
    # If rank 28 doesn't work, try a numerical search for lower ranks
    # Use alternating least squares to find a decomposition
    best_result = {"U": U_school.astype(int).tolist(), "V": V_school.astype(int).tolist(), "W": W_school.astype(int).tolist()}
    
    # Try ranks from 28 down to 20
    for target_rank in range(28, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            
            # Initialize with random values
            U_init = np.random.randn(size_a, target_rank)
            V_init = np.random.randn(size_b, target_rank)
            W_init = np.random.randn(size_c, target_rank)
            
            # Scale to get reasonable magnitudes
            U_init /= np.sqrt(size_a)
            V_init /= np.sqrt(size_b)
            W_init /= np.sqrt(size_c)
            
            # Alternating least squares iterations
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve min ||T[a,:,:] - sum_r U[a,r]*V[r,:]*W[r,:]||
                # This is a linear system: for each a, T[a,:,:] = U[a,:] @ (V * W)^T reshaped
                # More precisely: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # So T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r]).reshape(size_b, size_c)
                
                # Build the design matrix for U
                # For each r, the contribution to T[a,:,:] is V[:,r] * W[:,r] (outer product)
                # So we need: T[a,:,:] ≈ sum_r U[a,r] * M_r where M_r = outer(V[:,r], W[:,r])
                
                # Reshape T to (size_a, size_b * size_c)
                T_flat = T_true.reshape(size_a, -1)
                
                # Build M: (size_b * size_c, target_rank)
                M_U = np.zeros((size_b * size_c, target_rank))
                for r in range(target_rank):
                    M_U[:, r] = (V_curr[:, r] * W_curr[:, r]).reshape(-1)
                
                # Solve: T_flat[a,:] ≈ U_curr[a,:] @ M_U.T
                # i.e., T_flat = U_curr @ M_U.T
                # So U_curr = T_flat @ pinv(M_U.T)
                try:
                    U_new, residuals, rank_u, svals = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)
                    U_curr = U_new.T
                except:
                    break
                
                # Update V
                M_V = np.zeros((size_a * size_c, target_rank))
                for r in range(target_rank):
                    M_V[:, r] = (U_curr[:, r] * W_curr[:, r]).reshape(-1)
                
                T_flat_v = T_true.transpose(0, 2, 1).reshape(size_a * size_c, -1).T  # (size_b, size_a * size_c)
                # Actually: T[b,a,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed b: T[b,:,:] over (a,c) = sum_r V[b,r] * (U[:,r] * W[:,r])
                T_reshaped_v = T_true.transpose(1, 0, 2).reshape(size_b, -1)  # (size_b, size_a*size_c)
                
                try:
                    V_new, residuals, rank_v, svals = np.linalg.lstsq(M_V.T, T_reshaped_v.T, rcond=None)
                    V_curr = V_new.T
                except:
                    break
                
                # Update W
                M_W = np.zeros((size_a * size_b, target_rank))
                for r in range(target_rank):
                    M_W[:, r] = (U_curr[:, r] * V_curr[:, r]).reshape(-1)
                
                T_reshaped_w = T_true.transpose(2, 0, 1).reshape(size_c, -1)  # (size_c, size_a*size_b)
                
                try:
                    W_new, residuals, rank_w, svals = np.linalg.lstsq(M_W.T, T_reshaped_w.T, rcond=None)
                    W_curr = W_new.T
                except:
                    break
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr, optimize=True)
                error = np.max(np.abs(T_recon - T_true))
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                if verify(U_curr, V_curr, W_curr):
                    # Try to round to nearest half-integers
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        break
                    else:
                        # Use the unrounded solution
                        best_result = {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
                        break
        
        if best_result != {"U": U_school.astype(int).tolist(), "V": V_school.astype(int).tolist(), "W": W_school.astype(int).tolist()}:
            break
    
    return best_result
# EVOLVE-BLOCK-END
