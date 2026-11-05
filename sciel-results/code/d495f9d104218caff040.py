import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = 4, 5, 5  # 4x5 x 5x5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    # We'll construct a rank-19 decomposition for 5x5x5, then restrict to 4x5x5
    # The 5x5x5 multiplication: A(5x5) * B(5x5) = C(5x5)
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # U shape: (5*5, rank) = (25, rank)
    # V shape: (5*5, rank) = (25, rank)  
    # W shape: (5*5, rank) = (25, rank)
    # After restriction: zero out U entries where row index >= 4 (i.e., row 4 of A)
    # U restricted: (4*5, rank) = (20, rank)
    
    # BCLR rank-19 decomposition for 5x5x5
    # Define 19 terms. Each term: (u_sparsity, v_sparsity, w_sparsity)
    # where sparsity is list of (index, value)
    
    # Using the standard BCLR construction for 5x5
    # Partition: {0,1,2,3,4} into blocks
    # The 19 multiplications are defined as follows:
    
    terms_5x5 = []
    
    # Helper to add a term
    def add_5x5_term(u_dict, v_dict, w_dict):
        terms_5x5.append((u_dict, v_dict, w_dict))
    
    # BCLR 19-term algorithm for 5x5x5
    # M1: (A[0:2,0:2] + A[2:4,2:4] + A[4,4]) * (B[0:2,0:2] + B[2:4,2:4] + B[4,4])
    # This is getting complex. Let me use a known explicit list.
    
    # Here's the explicit BCLR rank-19 decomposition:
    # Each entry is (U_entries, V_entries, W_entries)
    # U index: i*5+j, V index: j*5+k, W index: k*5+i
    
    # Term 1
    add_5x5_term(
        {(0*5+0, 1), (1*5+1, 1), (2*5+2, 1), (3*5+3, 1), (4*5+4, 1)},
        {(0*5+0, 1), (1*5+1, 1), (2*5+2, 1), (3*5+3, 1), (4*5+4, 1)},
        {(0*5+0, 1), (1*5+1, 1), (2*5+2, 1), (3*5+3, 1), (4*5+4, 1)}
    )
    # This is just the diagonal, not BCLR. Let me use the actual BCLR formulas.
    
    # Let me use a different approach: I'll construct the 5x5x5 tensor, 
    # and use a known rank-19 decomposition from the literature.
    
    # Actually, let me just hardcode a verified rank-19 decomposition.
    # I'll use the one from the BCLR paper.
    
    # Clear and restart with proper BCLR terms
    terms_5x5.clear()
    
    # BCLR 19-term algorithm:
    # The 19 bilinear forms are:
    
    # Using block partition: rows/cols split as {0,1}, {2,3}, {4}
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # where A00 is 2x2, A01 is 2x2, A02 is 2x1, etc.
    
    # This is very complex to write out. Let me use a simpler approach:
    # I'll verify that restricting a known 5x5x5 rank-19 decomposition works.
    
    # Alternative: use the fact that we can embed 4x5x5 into 5x5x5
    # and use the restriction property.
    
    # For now, let me use the current program's approach (Strassen-based) 
    # but also try the BCLR restriction if I can get the terms.
    
    # Let me fall back to the Strassen-based approach which gives rank 92,
    # and see if I can improve it.
    
    # Actually, let me try a different partition that might give better rank.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    ig = i_off + i_loc
                    jg = j_off + j_loc
                    kg = k_off + k_loc
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
    
    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
        
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )
    
    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)
    
    # Try different partitions to minimize rank
    
    # Partition 1: n=[2,2], m=[2,3], p=[2,3]
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    rank = len(U_cols)
    
    # Build matrices
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        W = np.zeros((p_n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via linear dependency removal
    if time.time() - start_time < DEADLINE - 20:
        term_vecs = []
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            t_c = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            term_vecs.append(t_c.flatten())
        
        M = np.array(term_vecs)
        
        try:
            sv = np.linalg.svd(M, compute_uv=False)
            tol = sv[0] * 1e-10
            num_indep = int(np.sum(sv > tol))
            
            if num_indep < rank:
                Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
                independent_indices = piv[:num_indep]
                
                U_red = U[:, independent_indices]
                V_red = V[:, independent_indices]
                W_red = W[:, independent_indices]
                
                U3r = U_red.reshape(n, m, num_indep)
                V3r = V_red.reshape(m, p, num_indep)
                W3r = W_red.reshape(p, n, num_indep)
                T_red = np.einsum('ijc,jkc,kic->ijk', U3r, V3r, W3r)
                
                if np.allclose(T_red, 1.0, atol=1e-9):
                    U = U_red
                    V = V_red
                    W = W_red
                    rank = num_indep
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
