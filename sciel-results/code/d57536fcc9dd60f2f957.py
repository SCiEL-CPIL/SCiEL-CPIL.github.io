import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Target tensor: all ones
    target = np.ones((n, m, p))
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        U_r = U.T.reshape(n, m, rank)
        V_r = V.T.reshape(m, p, rank)
        W_r = W.T.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(result - target)) < tol
    
    def als_optimize(U_init, V_init, W_init, n, m, p, target_rank, max_iters=200, tol=1e-10, deadline=None):
        """Alternating Least Squares for tensor decomposition"""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: fix V and W
            # For each column c of U, we need to minimize sum_{i,j} (target[i,j,:] - sum_c U[i,j,c] V[j,:,c] W[:,i,c])^2
            # This is linear in U[:, c] for fixed V, W
            
            # Precompute for efficiency
            # tmp[j, k, c] = V[j, k, c] * W[k, i, c] -- but this depends on i
            # Let's think differently.
            
            # The objective is: sum_{i,j,k} (T[i,j,k] - sum_c U_c[i,j] V_c[j,k] W_c[k,i])^2
            # For fixed V, W, this is quadratic in U.
            
            # Let's use a different approach: treat each (i,j) pair as a separate variable
            # U has shape (n*m, r), so U[i*m+j, c] is the variable
            
            # Actually, let's use the fact that for fixed V and W:
            # The contribution of U[i*m+j, c] to the error is:
            # sum_k (T[i,j,k] - sum_c' U[i*m+j, c'] V_c'[j,k] W_c'[k,i])^2
            # = sum_k (T[i,j,k])^2 - 2 sum_k T[i,j,k] sum_c' U[i*m+j, c'] V_c'[j,k] W_c'[k,i] + sum_k (sum_c' U[i*m+j, c'] V_c'[j,k] W_c'[k,i])^2
            
            # This is getting complex. Let's use a simpler ALS approach.
            
            # Alternative: use the tensor network structure
            # For fixed V, W: minimize over U
            # Define A[i,j,k] = sum_c V_c[j,k] W_c[k,i]
            # Then we want: sum_k (T[i,j,k] - sum_c U[i*m+j, c] A_c[i,j,k])^2
            # Wait, A depends on c. Let me re-derive.
            
            # sum_k (T[i,j,k] - sum_c U[i*m+j, c] V_c[j,k] W_c[k,i])^2
            # Let B_c[i,j,k] = V_c[j,k] W_c[k,i]
            # Then: sum_k (T[i,j,k] - sum_c U[i*m+j, c] B_c[i,j,k])^2
            
            # This is a least squares problem in U[i*m+j, :]:
            # min_x sum_k (T[i,j,k] - x^T B_[:,i,j,k])^2
            # where x = U[i*m+j, :] and B_[:,i,j,k] = [B_0[i,j,k], B_1[i,j,k], ..., B_{r-1}[i,j,k]]
            
            # So for each (i,j), we solve:
            # min_x ||T[i,j,:] - B_{i,j}^T x||^2
            # where B_{i,j} has shape (r, p) with B_{i,j}[c, k] = V_c[j,k] W_c[k,i]
            
            # Solution: x = (B_{i,j} B_{i,j}^T)^{-1} B_{i,j} T[i,j,:]
            # Or use lstsq for numerical stability
            
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # Build B_{i,j}: shape (r, p)
                    # B[c, k] = V_c[j, k] * W_c[k, i]
                    # V_c[j, k] = V[j*p+k, c]
                    # W_c[k, i] = W[k*n+i, c]
                    
                    # V_j = V[j*p:(j+1)*p, :]  # shape (p, r)
                    # W_i = W[:, i::n]  # shape (p, r) -- W is (p*n, r), W[k*n+i, c]
                    # Actually W is indexed as W[k*n + i, c], so W_i[k, c] = W[k*n + i, c]
                    
                    V_j = V[j*p:(j+1)*p, :]  # (p, r)
                    W_i = W.reshape(p, n, target_rank)[:, i, :]  # (p, r)
                    
                    # B[c, k] = V_j[k, c] * W_i[k, c]
                    B = (V_j.T * W_i.T).T  # (r, p) -- B[c, k] = V_j[k,c] * W_i[k,c]
                    
                    # Solve: min_x ||T[i,j,:] - B^T x||^2
                    # B^T has shape (p, r)
                    # x has shape (r,)
                    # Use lstsq
                    try:
                        x, _, _, _ = np.linalg.lstsq(B.T, target[i, j, :], rcond=None)
                        U_new[i*m + j, :] = x
                    except:
                        pass
            
            U = U_new
            
            # Update V: fix U and W
            # Similarly, for each (j,k):
            # min_y ||T[:,j,k] - C_{j,k}^T y||^2
            # where C_{j,k}[c, i] = U_c[i,j] * W_c[k,i]
            
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    # C[c, i] = U_c[i, j] * W_c[k, i]
                    # U_c[i, j] = U[i*m + j, c]
                    # W_c[k, i] = W[k*n + i, c]
                    
                    U_j = U[:, j::m]  # (n, r) -- U[i*m+j, c] for all i
                    # Actually U is (n*m, r), U[i*m+j, c]. So U[:, j::m] doesn't work directly.
                    # U.reshape(n, m, r)[:, j, :] gives (n, r)
                    U_j = U.reshape(n, m, target_rank)[:, j, :]  # (n, r)
                    W_k = W.reshape(p, n, target_rank)[k, :, :]  # (n, r)
                    
                    # C[c, i] = U_j[i, c] * W_k[i, c]
                    C = (U_j.T * W_k.T).T  # (r, n) -- C[c, i] = U_j[i,c] * W_k[i,c]
                    
                    # Solve: min_y ||T[:,j,k] - C^T y||^2
                    # T[:,j,k] has shape (n,)
                    try:
                        y, _, _, _ = np.linalg.lstsq(C.T, target[:, j, k], rcond=None)
                        V_new[j*p + k, :] = y
                    except:
                        pass
            
            V = V_new
            
            # Update W: fix U and V
            # For each (k,i):
            # min_z ||T[:, :, k] - D_{k,i}^T z||^2
            # where D_{k,i}[c, j] = U_c[i,j] * V_c[j,k]
            
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    # D[c, j] = U_c[i, j] * V_c[j, k]
                    # U_c[i, j] = U[i*m + j, c]
                    # V_c[j, k] = V[j*p + k, c]
                    
                    U_i = U[i*m:(i+1)*m, :]  # (m, r) -- U[i*m+j, c]
                    V_k = V[:, k::p]  # (m, r) -- V[j*p+k, c] for all j
                    # Actually V is (m*p, r), V[j*p+k, c]. So V.reshape(m, p, r)[:, k, :] gives (m, r)
                    V_k = V.reshape(m, p, target_rank)[:, k, :]  # (m, r)
                    
                    # D[c, j] = U_i[j, c] * V_k[j, c]
                    D = (U_i.T * V_k.T).T  # (r, m) -- D[c, j] = U_i[j,c] * V_k[j,c]
                    
                    # Solve: min_z ||T[:,:,k] - D^T z||^2
                    # T[:,:,k] has shape (n, m)
                    # D^T has shape (m, r)
                    # We need: min_z ||T[:,:,k].flatten() - (D^T ⊗ I) z||^2
                    # Or: min_z ||T[:,:,k] - D^T z||_F^2
                    # This is: min_z sum_{i,j} (T[i,j,k] - sum_c D[c,j] z[c])^2
                    # = min_z sum_j (T[:,j,k] - D[:,j]^T z)^2
                    # Stack: min_z ||T[:,:,k].T.flatten() - (D.T.T @ I) z||^2
                    # Actually: min_z ||T[:,:,k] - D.T @ z||_F^2
                    # Let A = D.T (m x r), b = T[:,:,k] (n x m)
                    # min_z ||b - A @ z||_F^2
                    # = min_z sum_i ||b[i,:] - A @ z||^2
                    # = min_z ||b.flatten() - (I_n ⊗ A) z||^2
                    # Use: z = (A.T @ A)^{-1} A.T @ b.T ... no
                    # Actually: min_z ||B - A z||_F^2 where B is (n,m), A is (m,r), z is (r,)
                    # This is: min_z sum_i ||B[i,:] - A z||^2
                    # Gradient: -2 A.T (B - A z) = 0 -> A.T A z = A.T B
                    # Wait: sum_i (B[i,:] - A z)^T (B[i,:] - A z)
                    # = sum_i (B[i,:]^T B[i,:] - 2 B[i,:]^T A z + z^T A^T A z)
                    # Gradient: -2 sum_i A^T B[i,:] + 2 A^T A z = 0
                    # A^T A z = A^T sum_i B[i,:] = A^T B.sum(axis=0)
                    # So z = (A^T A)^{-1} A^T B.sum(axis=0)
                    
                    A = D.T  # (m, r)
                    B = target[:, :, k]  # (n, m)
                    b_sum = B.sum(axis=0)  # (m,)
                    
                    try:
                        # Solve A.T @ A @ z = A.T @ b_sum
                        ATA = A.T @ A  # (r, r)
                        ATb = A.T @ b_sum  # (r,)
                        z, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
                        W_new[k*n + i, :] = z
                    except:
                        pass
            
            W = W_new
            
            # Compute error
            U_r = U.T.reshape(n, m, target_rank)
            V_r = V.T.reshape(m, p, target_rank)
            W_r = W.T.reshape(p, n, target_rank)
            result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = np.sum((result - target) ** 2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Final error
        U_r = U.T.reshape(n, m, target_rank)
        V_r = V.T.reshape(m, p, target_rank)
        W_r = W.T.reshape(p, n, target_rank)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = np.sum((result - target) ** 2)
        
        return U, V, W, err
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try ranks from 60 to 74
    for target_rank in range(60, 75):
        if time.time() > deadline - 15:
            break
        
        # Try multiple initializations
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            
            # Different initialization strategies
            if seed == 0:
                # Random
                U_init = np.random.randn(n * m, target_rank) * 0.5
                V_init = np.random.randn(m * p, target_rank) * 0.5
                W_init = np.random.randn(p * n, target_rank) * 0.5
            elif seed == 1:
                # Schoolbook subset
                U_init = U_school[:, :target_rank].copy()
                V_init = V_school[:, :target_rank].copy()
                W_init = W_school[:, :target_rank].copy()
            elif seed == 2:
                # Random + schoolbook mix
                U_init = U_school[:, :target_rank].copy() * 0.5 + np.random.randn(n * m, target_rank) * 0.3
                V_init = V_school[:, :target_rank].copy() * 0.5 + np.random.randn(m * p, target_rank) * 0.3
                W_init = W_school[:, :target_rank].copy() * 0.5 + np.random.randn(p * n, target_rank) * 0.3
            elif seed == 3:
                # Structured: block diagonal-ish
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U_init[i * m + j, c] = 1
                    V_init[j * p + k, c] = 1
                    W_init[k * n + i, c] = 1
            else:
                # Scaled random
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            
            try:
                U_opt, V_opt, W_opt, err = als_optimize(
                    U_init, V_init, W_init, n, m, p, target_rank,
                    max_iters=100, tol=1e-12, deadline=deadline - 5
                )
                
                if err < 1e-6:
                    if verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                            found = True
                            break
            except Exception:
                continue
        
        if found:
            # Found a valid decomposition, try lower ranks
            continue
    
    # Final polish: try to snap to nice values
    if best_rank < rank_schoolbook:
        U_final = np.array(best_result["U"])
        V_final = np.array(best_result["V"])
        W_final = np.array(best_result["W"])
        
        # Try rounding to half-integers
        U_rounded = np.round(U_final * 2) / 2
        V_rounded = np.round(V_final * 2) / 2
        W_rounded = np.round(W_final * 2) / 2
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-4):
            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
