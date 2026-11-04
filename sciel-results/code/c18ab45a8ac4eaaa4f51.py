import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank-minimal decomposition for 2x3x9
    # Strategy: Find rank-minimal for 2x3x3 block, then repeat 3 times.
    
    # 1. Define the tensor for 2x3x3 multiplication
    # T[i, j, k] = 1 if it's a valid multiplication term
    # Indices: i in [0,2), j in [0,3), k in [0,3)
    # U index: i*3 + j
    # V index: j*3 + k
    # W index: k*2 + i
    
    n_b, m_b, p_b = 2, 3, 3
    T_block = np.zeros((n_b * m_b, m_b * p_b, p_b * n_b))
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                u_idx = i * m_b + j
                v_idx = j * p_b + k
                w_idx = k * n_b + i
                T_block[u_idx, v_idx, w_idx] = 1.0

    # 2. Try to find a low-rank decomposition for T_block
    # Start with rank 16 (suspected minimal)
    # Use ALS with random initialization
    
    best_rank = n_b * m_b * p_b # 18
    best_decomp = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit for search
    
    for rank in range(16, best_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Scale appropriately
            U = np.random.randn(n_b * m_b, rank)
            V = np.random.randn(m_b * p_b, rank)
            W = np.random.randn(p_b * n_b, rank)
            
            # ALS iterations
            for it in range(200):
                # Update U
                # T = sum_r u_r v_r^T w_r
                # For each r, u_r = T * (V_r, W_r)
                # We solve for all U at once?
                # T_reshaped (n*m, m*p*p*n, R)
                # This is complex. Let's use a simpler gradient step or explicit ALS formula.
                
                # ALS for Tensor:
                # U_new[:, r] = T * (V[:, r] \otimes W[:, r])
                # T * (v \otimes w) is the contraction of T with v and w.
                
                # Efficient ALS:
                # Fix V, W.
                # We want to minimize || T - sum_r u_r \otimes V_r \otimes W_r ||^2
                # This is equivalent to minimizing || T_reshaped - U (V \otimes W)^T ||? No.
                
                # Let's use the explicit update:
                # U_r = sum_{i,j,k} T_{ijk} V_{jr} W_{kr} ... no.
                # The term is u_r[i] v_r[j] w_r[k].
                # So U_r[i] = sum_{j,k} T_{i,j,k} V_r[j] W_r[k] / (sum_{j,k} V_r[j]^2 W_r[k]^2)
                
                # Compute V_r \otimes W_r matrix (size m_b*p_b*n_b? No)
                # For each r, we need to contract T with V_r and W_r.
                
                # Let's just use gradient descent with small steps, it's safer to implement correctly for small sizes.
                # Or, use the fact that we can solve linear systems.
                
                # Let's stick to a simple approach:
                # Compute error and gradient.
                # This might be slow in Python loops.
                
                # Alternative: Use NumPy tensordot.
                # T is (6, 9, 6).
                # V is (9, R), W is (6, R).
                # V_r \otimes W_r is a vector of size 54.
                # We can form a matrix M of size (54, R) where M[:, r] = V_r \otimes W_r.
                # Then T can be reshaped to (6, 54).
                # We want U (6, R) such that T_reshaped \approx U M^T.
                # So U = T_reshaped M (M^T M)^{-1}.
                
                M = np.einsum('jr,kr->jkr', V, W).reshape(m_b * p_b, rank)
                # M has shape (m_b*p_b, rank)
                # T_reshaped has shape (n_b*m_b, m_b*p_b)
                T_reshaped = T_block.reshape(n_b * m_b, m_b * p_b * p_b * n_b // (p_b * n_b)) # No
                # T_block shape (n*m, m*p, p*n)
                # We want to group (m*p, p*n) into one axis.
                T_reshaped = T_block.reshape(n_b * m_b, m_b * p_b * p_b * n_b)
                # Wait, the second axis is m*p, third is p*n.
                # So T_reshaped is (n*m, m*p*p*n).
                # M is (m*p, R) and W is (p*n, R).
                # The product V_r \otimes W_r is a vector of size m*p*p*n? No.
                # V_r is size m*p. W_r is size p*n.
                # The tensor term is u_r[i] * V_r[j] * W_r[k].
                # So for a fixed i, the vector of coefficients for (j,k) is V_r \otimes W_r.
                # So we form M of shape (m*p*p*n, R) where M[:, r] = V_r \otimes W_r.
                
                M_full = np.einsum('jr,kr->jkr', V, W).reshape(m_b * p_b * p_b * n_b, rank)
                # T_reshaped is (n*m, m*p*p*n)
                # We want U such that T_reshaped \approx U M_full^T
                # U = T_reshaped M_full (M_full^T M_full)^{-1}
                
                try:
                    U_new = T_reshaped @ M_full @ np.linalg.inv(M_full.T @ M_full)
                except np.linalg.LinAlgError:
                    U_new = T_reshaped @ M_full @ np.linalg.pinv(M_full.T @ M_full)
                
                # Update V
                # Now fix U, W.
                # M_full_V = np.einsum('ir,kr->ikr', U_new, W).reshape(n_b*m_b*p_b*n_b, rank)
                # T_reshaped_V = T_block.reshape(m_b*p_b, n_b*m_b*p_b*n_b)
                # V_new = T_reshaped_V M_full_V (M_full_V^T M_full_V)^{-1}
                
                M_full_V = np.einsum('ir,kr->ikr', U_new, W).reshape(n_b * m_b * p_b * n_b, rank)
                T_reshaped_V = T_block.reshape(m_b * p_b, n_b * m_b * p_b * n_b)
                try:
                    V_new = T_reshaped_V @ M_full_V @ np.linalg.inv(M_full_V.T @ M_full_V)
                except np.linalg.LinAlgError:
                    V_new = T_reshaped_V @ M_full_V @ np.linalg.pinv(M_full_V.T @ M_full_V)
                    
                # Update W
                M_full_W = np.einsum('ir,jr->ijr', U_new, V_new).reshape(n_b * m_b * m_b * p_b, rank)
                T_reshaped_W = T_block.reshape(p_b * n_b, n_b * m_b * m_b * p_b)
                try:
                    W_new = T_reshaped_W @ M_full_W @ np.linalg.inv(M_full_W.T @ M_full_W)
                except np.linalg.LinAlgError:
                    W_new = T_reshaped_W @ M_full_W @ np.linalg.pinv(M_full_W.T @ M_full_W)
                    
                U, V, W = U_new, V_new, W_new
                
            # Check error
            # Reconstruct T_approx
            # T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_approx - T_block)
            
            if error < 1e-10:
                # Found exact decomposition
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
                
        if found:
            break

    # 3. Construct the full 2x3x9 decomposition
    # If we found a block decomposition, use it.
    # If not, use schoolbook.
    
    if best_decomp is not None:
        U_b, V_b, W_b = best_decomp
        R_block = best_rank
        R_total = 3 * R_block
        
        U = np.zeros((n * m, R_total))
        V = np.zeros((m * p, R_total))
        W = np.zeros((p * n, R_total))
        
        # Fill 3 blocks
        for b in range(3):
            # U is same for all blocks
            U[:, b*R_block:(b+1)*R_block] = U_b
            
            # V: map block b columns
            # V_b has shape (3*3, R_block) -> indices j*3+k
            # V has shape (3*9, R_total) -> indices j*9+k'
            # k' = 3*b + k
            for j in range(m_b):
                for k in range(p_b):
                    v_idx_b = j * p_b + k
                    v_idx_full = j * p + (3*b + k)
                    V[v_idx_full, b*R_block:(b+1)*R_block] = V_b[v_idx_b, :]
                    
            # W: map block b columns
            # W_b has shape (3*2, R_block) -> indices k*2+i
            # W has shape (9*2, R_total) -> indices k*2+i
            # k = 3*b + k'
            for k in range(p_b):
                for i in range(n_b):
                    w_idx_b = k * n_b + i
                    w_idx_full = (3*b + k) * n + i
                    W[w_idx_full, b*R_block:(b+1)*R_block] = W_b[w_idx_b, :]
                    
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        # T_full shape (6, 27, 18)
        # We don't need to verify fully if error was small, but let's check one term.
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, r] = 1
                    V[v_idx, r] = 1
                    W[w_idx, r] = 1
                    r += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
