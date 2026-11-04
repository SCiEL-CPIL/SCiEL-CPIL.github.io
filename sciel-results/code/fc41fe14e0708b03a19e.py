# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Strategy: Decompose into 3 blocks of 2x3 by 3x3
    # Each block has rank 15 (known result). Total rank 45.
    # We will try to find the rank 15 decomposition for 2x3 by 3x3.
    
    n_block, m_block, p_block = 2, 3, 3
    dim_u_b = n_block * m_block  # 6
    dim_v_b = m_block * p_block  # 9
    dim_w_b = p_block * n_block  # 6
    target_rank_b = 15
    
    def find_block_decomposition(n_b, m_b, p_b, rank_b, max_iters=200, trials=10):
        """Find a rank-rank_b decomposition for n_b x m_b by m_b x p_b."""
        dim_u_b = n_b * m_b
        dim_v_b = m_b * p_b
        dim_w_b = p_b * n_b
        
        # Precompute target tensor indices
        # T[a, b, c] = 1 if a=i*m_b+j, b=j*p_b+k, c=k*n_b+i
        # We store non-zero entries and zero entries
        # Actually, we just need the vector T_a for each a, T_b for each b, T_c for each c
        # T_a is a vector of length dim_v_b * dim_w_b
        # T_b is a vector of length dim_u_b * dim_w_b
        # T_c is a vector of length dim_u_b * dim_v_b
        
        T_vectors_a = []
        T_vectors_b = []
        T_vectors_c = []
        
        # Initialize with zeros
        T_a_arr = np.zeros((dim_u_b, dim_v_b * dim_w_b))
        T_b_arr = np.zeros((dim_v_b, dim_u_b * dim_w_b))
        T_c_arr = np.zeros((dim_w_b, dim_u_b * dim_v_b))
        
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a = i * m_b + j
                    b = j * p_b + k
                    c = k * n_b + i
                    
                    # Fill T_a
                    idx_a = b * dim_w_b + c
                    T_a_arr[a, idx_a] = 1.0
                    
                    # Fill T_b
                    idx_b = a * dim_w_b + c
                    T_b_arr[b, idx_b] = 1.0
                    
                    # Fill T_c
                    idx_c = a * dim_v_b + b
                    T_c_arr[c, idx_c] = 1.0
        
        for trial in range(trials):
            if time.time() > deadline - 10:
                return None
            
            rng = np.random.RandomState(trial)
            U = rng.randn(dim_u_b, rank_b) * 0.5
            V = rng.randn(dim_v_b, rank_b) * 0.5
            W = rng.randn(dim_w_b, rank_b) * 0.5
            
            # Alternating minimization (Vectorized)
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U
                # K[b, c, d] = V[b, d] * W[c, d]
                # K_mat shape: (dim_v_b * dim_w_b, rank_b)
                # K_mat[(b*dim_w_b+c), d] = V[b, d] * W[c, d]
                K_mat = np.zeros((dim_v_b * dim_w_b, rank_b))
                for b in range(dim_v_b):
                    for c in range(dim_w_b):
                        idx = b * dim_w_b + c
                        K_mat[idx, :] = V[b, :] * W[c, :]
                
                # Solve K_mat @ U[a, :] = T_a_arr[a, :] for each a
                U_new = np.zeros_like(U)
                for a in range(dim_u_b):
                    try:
                        U_new[a, :], _, _, _ = np.linalg.lstsq(K_mat, T_a_arr[a, :], rcond=None)
                    except:
                        continue
                U = U_new
                
                # Solve for V
                # M[a, c, d] = U[a, d] * W[c, d]
                # M_mat shape: (dim_u_b * dim_w_b, rank_b)
                M_mat = np.zeros((dim_u_b * dim_w_b, rank_b))
                for a in range(dim_u_b):
                    for c in range(dim_w_b):
                        idx = a * dim_w_b + c
                        M_mat[idx, :] = U[a, :] * W[c, :]
                
                V_new = np.zeros_like(V)
                for b in range(dim_v_b):
                    try:
                        V_new[b, :], _, _, _ = np.linalg.lstsq(M_mat, T_b_arr[b, :], rcond=None)
                    except:
                        continue
                V = V_new
                
                # Solve for W
                # N[a, b, d] = U[a, d] * V[b, d]
                # N_mat shape: (dim_u_b * dim_v_b, rank_b)
                N_mat = np.zeros((dim_u_b * dim_v_b, rank_b))
                for a in range(dim_u_b):
                    for b in range(dim_v_b):
                        idx = a * dim_v_b + b
                        N_mat[idx, :] = U[a, :] * V[b, :]
                
                W_new = np.zeros_like(W)
                for c in range(dim_w_b):
                    try:
                        W_new[c, :], _, _, _ = np.linalg.lstsq(N_mat, T_c_arr[c, :], rcond=None)
                    except:
                        continue
                W = W_new
                
                # Check error
                err = 0.0
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            a = i*m_b+j
                            b_idx = j*p_b+k
                            c_idx = k*n_b+i
                            val = np.sum(U[a,:] * V[b_idx,:] * W[c_idx,:])
                            err += (val - 1)**2
                # Sample zero entries
                rng_check = np.random.RandomState(it)
                for _ in range(50):
                    a = rng_check.randint(dim_u_b)
                    b = rng_check.randint(dim_v_b)
                    c = rng_check.randint(dim_w_b)
                    i = a // m_b
                    j = a % m_b
                    j2 = b // p_b
                    k = b % p_b
                    k2 = c // n_b
                    i2 = c % n_b
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        err += val**2
                
                if err < 1e-8:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_r = 0.0
                    for i in range(n_b):
                        for j in range(m_b):
                            for k in range(p_b):
                                a = i*m_b+j
                                b_idx = j*p_b+k
                                c_idx = k*n_b+i
                                val = np.sum(U_r[a,:] * V_r[b_idx,:] * W_r[c_idx,:])
                                err_r += (val - 1)**2
                    # Check all zeros for safety in small case
                    for a in range(dim_u_b):
                        for b in range(dim_v_b):
                            for c in range(dim_w_b):
                                i = a // m_b
                                j = a % m_b
                                j2 = b // p_b
                                k = b % p_b
                                k2 = c // n_b
                                i2 = c % n_b
                                if not (j == j2 and k == k2 and i == i2):
                                    val = np.sum(U_r[a,:] * V_r[b,:] * W_r[c,:])
                                    err_r += val**2
                            
                    if err_r < 1e-8:
                        return U_r, V_r, W_r
        
        return None
    
    # Try to find the block decomposition
    block_decomp = find_block_decomposition(n_block, m_block, p_block, target_rank_b)
    
    if block_decomp is not None:
        U_b, V_b, W_b = block_decomp
        # Construct full decomposition
        # p = 9 = 3 * 3
        # We have 3 blocks
        full_rank = target_rank_b * 3
        U_full = np.zeros((dim_u, full_rank))
        V_full = np.zeros((dim_v, full_rank))
        W_full = np.zeros((dim_w, full_rank))
        
        for block_idx in range(3):
            k_offset = block_idx * p_block
            for c in range(target_rank_b):
                full_c = block_idx * target_rank_b + c
                # U is same for all blocks
                U_full[:, full_c] = U_b[:, c]
                
                # V: indices j*p + k
                for j in range(m_block):
                    for k in range(p_block):
                        v_block_idx = j * p_block + k
                        v_full_idx = j * p + (k_offset + k)
                        V_full[v_full_idx, full_c] = V_b[v_block_idx, c]
                        
                # W: indices k*n + i
                for k in range(p_block):
                    for i in range(n_block):
                        w_block_idx = k * n_block + i
                        w_full_idx = (k_offset + k) * n + i
                        W_full[w_full_idx, full_c] = W_b[w_block_idx, c]
        
        # Verify full decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U_full[a,:] * V_full[b,:] * W_full[c,:])
                    if abs(val - 1) > 1e-8:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Check zero entries (sample)
            rng = np.random.RandomState(42)
            for _ in range(100):
                a = rng.randint(dim_u)
                b = rng.randint(dim_v)
                c = rng.randint(dim_w)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    val = np.sum(U_full[a,:] * V_full[b,:] * W_full[c,:])
                    if abs(val) > 1e-8:
                        valid = False
                        break
            
            if valid:
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = full_rank
    
    return best_result
# EVOLVE-BLOCK-END
