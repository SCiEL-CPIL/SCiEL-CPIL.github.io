import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    
    # Target: Rank 14 (2 * Rank 7 for 2x3x3)
    # We will try to find a rank 7 decomposition for 2x3x3
    # and use it twice.
    
    # Tensor for 2x3x3
    n1, m1, p1 = 2, 3, 3
    T1 = np.zeros((n1*m1, m1*p1, p1*n1))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                a = i*m1 + j
                b = j*p1 + k
                c = k*n1 + i
                T1[a, b, c] = 1.0
                
    # Try to find rank 7 decomposition
    rank1 = 7
    U1 = None
    V1 = None
    W1 = None
    
    # Simple random search / ALS
    # This is a heuristic. If it fails, we fallback.
    # Given the small size, we can try a few restarts.
    
    start_time = time.time()
    time_limit = 10.0
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U_rand = np.random.randn(n1*m1, rank1)
        V_rand = np.random.randn(m1*p1, rank1)
        W_rand = np.random.randn(p1*n1, rank1)
        
        # Scale to avoid overflow
        U_rand /= np.linalg.norm(U_rand)
        V_rand /= np.linalg.norm(V_rand)
        W_rand /= np.linalg.norm(W_rand)
        
        # ALS iterations
        for _ in range(200):
            # Update U
            # T1 = sum U_t V_t W_t
            # U_t = argmin ||T1 - sum U_t V_t W_t||
            # This is linear in U_t if V_t, W_t fixed.
            # T1[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # For fixed t, U[:,t] is a vector.
            # We can solve for all U simultaneously?
            # Let's just do gradient descent or simple alternating least squares.
            
            # ALS for U:
            # Minimize || T1 - sum_t U[:,t] outer(V[:,t], W[:,t]) ||^2
            # This is equivalent to solving a linear system for each t?
            # No, U[:,t] is independent.
            # Let M_t = outer(V[:,t], W[:,t]).
            # We want T1_approx = sum_t U[:,t] M_t.
            # This is not a standard least squares because U[:,t] multiplies M_t element-wise?
            # No, U[:,t] is a vector of length n1*m1.
            # T1 is a tensor.
            # T1[a,b,c] = sum_t U[a,t] V[b,t] W[c,t].
            # For a fixed t, the term is U[:,t] * (V[:,t] W[:,t]^T) ? No.
            # It's U[:,t] \otimes V[:,t] \otimes W[:,t].
            # So U[:,t] is just a vector.
            # The residual R = T1 - sum_{t' != t} U[:,t'] V[:,t'] W[:,t'].
            # We want to minimize || R - U[:,t] V[:,t] W[:,t] ||.
            # This is not linear in U[:,t] because V and W are fixed vectors?
            # Yes, it is linear in U[:,t] if we view the tensor as a matrix?
            # Let's reshape T1 to (n1*m1, m1*p1*p1*n1).
            # Then T1_flat[a, idx] = sum_t U[a,t] V[b,t] W[c,t].
            # This is still not a simple matrix product.
            
            # Let's use a simple gradient descent.
            # Or just use the fact that we can solve for U if we fix V and W?
            # T1[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Let B_t[b,c] = V[b,t] W[c,t].
            # Then T1[a,:,:] = sum_t U[a,t] B_t.
            # This is a linear system for U[a,t].
            # For each a, T1[a,:,:] is a matrix of size (m1*p1, p1*n1).
            # We want to represent it as sum_t U[a,t] B_t.
            # This is a least squares problem for each a.
            # We can solve for U[:,t] for all t simultaneously?
            # Let's stack B_t into a matrix B_mat of size (m1*p1*p1*n1, rank1).
            # B_mat[idx, t] = V[b,t] W[c,t] where idx = b*(p1*n1) + c.
            # Then T1_flat (size n1*m1 x m1*p1*p1*n1) = U (size n1*m1 x rank1) @ B_mat.T?
            # T1_flat[a, idx] = sum_t U[a,t] B_mat[idx, t].
            # Yes!
            # So U = T1_flat @ B_mat.T @ (B_mat @ B_mat.T)^-1 ?
            # U = T1_flat @ (B_mat.T B_mat)^-1 B_mat.T ?
            # Let's compute B_mat.
            
            B_mat = np.zeros((m1*p1*p1*n1, rank1))
            for t in range(rank1):
                B_t = np.outer(V_rand[:,t], W_rand[:,t])
                B_mat[:,t] = B_t.flatten()
                
            # Solve U = T1_flat @ B_mat.T @ pinv(B_mat @ B_mat.T)
            # Or simpler: U = lstsq(B_mat.T, T1_flat.T).T ?
            # T1_flat = U B_mat.T
            # U = T1_flat B_mat (B_mat.T B_mat)^-1
            
            try:
                # U (n1*m1, rank1) = T1_flat (n1*m1, M) @ B_mat.T (M, rank1) @ inv(B_mat @ B_mat.T) (rank1, rank1)
                # This is expensive.
                # Let's use np.linalg.lstsq(B_mat.T, T1_flat.T).T
                U_sol, _, _, _ = np.linalg.lstsq(B_mat.T, T1_flat.T, rcond=None)
                U_rand = U_sol.T
            except:
                pass
                
            # Similarly for V and W
            
            # Update V
            # T1[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Let A_t[a,c] = U[a,t] W[c,t].
            # T1[:,b,:] = sum_t V[b,t] A_t.
            # A_mat (m1*p1*p1*n1, rank1) where A_mat[idx, t] = U[a,t] W[c,t].
            # T1_flat (n1*m1, M) -> We need to reshape T1 to (m1*p1, n1*m1*p1*n1)?
            # T1[b, a, c] = sum_t V[b,t] U[a,t] W[c,t].
            # Let's permute T1 to (b, a, c).
            T1_perm = np.transpose(T1, (1, 0, 2)) # (m1*p1, n1*m1, p1*n1)
            T1_perm_flat = T1_perm.reshape(m1*p1, -1)
            
            A_mat = np.zeros((n1*m1*p1*n1, rank1))
            for t in range(rank1):
                A_t = np.outer(U_rand[:,t], W_rand[:,t])
                A_mat[:,t] = A_t.flatten()
                
            try:
                V_sol, _, _, _ = np.linalg.lstsq(A_mat.T, T1_perm_flat.T, rcond=None)
                V_rand = V_sol.T
            except:
                pass
                
            # Update W
            # T1[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Let C_t[a,b] = U[a,t] V[b,t].
            # T1[:,:,c] = sum_t W[c,t] C_t.
            # Permute T1 to (c, a, b).
            T1_perm2 = np.transpose(T1, (2, 0, 1)) # (p1*n1, n1*m1, m1*p1)
            T1_perm2_flat = T1_perm2.reshape(p1*n1, -1)
            
            C_mat = np.zeros((n1*m1*m1*p1, rank1))
            for t in range(rank1):
                C_t = np.outer(U_rand[:,t], V_rand[:,t])
                C_mat[:,t] = C_t.flatten()
                
            try:
                W_sol, _, _, _ = np.linalg.lstsq(C_mat.T, T1_perm2_flat.T, rcond=None)
                W_rand = W_sol.T
            except:
                pass
                
        # Check error
        recon = np.zeros_like(T1)
        for t in range(rank1):
            recon += np.einsum('a,b,c->abc', U_rand[:,t], V_rand[:,t], W_rand[:,t])
        err = np.linalg.norm(recon - T1)
        
        if err < 1e-6:
            U1, V1, W1 = U_rand, V_rand, W_rand
            break
            
    if U1 is not None:
        # Construct 2x3x6 decomposition
        # Block 1: k=0,1,2 (indices 0..8 in V, 0..5 in W)
        # Block 2: k=3,4,5 (indices 9..17 in V, 6..11 in W)
        
        rank_total = 14
        U_total = np.zeros((n*m, rank_total))
        V_total = np.zeros((m*p, rank_total))
        W_total = np.zeros((p*n, rank_total))
        
        # Copy Block 1
        U_total[:, 0:7] = U1
        V_total[0:9, 0:7] = V1
        W_total[0:6, 0:7] = W1
        
        # Copy Block 2 (shifted indices)
        U_total[:, 7:14] = U1
        # V indices for k=3,4,5 are j*6 + k.
        # V1 has indices j*3 + k for k=0,1,2.
        # We need to map V1[b] to V_total[b + 3*?].
        # b = j*3 + k.
        # New b' = j*6 + (k+3) = j*6 + k + 3 = 2*(j*3 + k) + 3? No.
        # b' = j*6 + k + 3.
        # Let's just iterate.
        for j in range(3):
            for k in range(3):
                old_b = j*3 + k
                new_b = j*6 + (k+3)
                V_total[new_b, 7:14] = V1[old_b, :]
                
        # W indices for k=3,4,5 are i*2 + k? No, W index is k*n + i.
        # W1 has indices k*2 + i for k=0,1,2.
        # New W index for k'=k+3 is (k+3)*2 + i = 2k + 6 + i.
        for k in range(3):
            for i in range(2):
                old_c = k*2 + i
                new_c = (k+3)*2 + i
                W_total[new_c, 7:14] = W1[old_c, :]
                
        # Verify
        recon_total = np.zeros((n*m, m*p, p*n))
        for t in range(rank_total):
            recon_total += np.einsum('a,b,c->abc', U_total[:,t], V_total[:,t], W_total[:,t])
        
        T_total = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_total[a, b, c] = 1.0
                    
        if np.linalg.norm(recon_total - T_total) < 1e-5:
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
            
    # Fallback: Schoolbook
    rank = n*m*p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
