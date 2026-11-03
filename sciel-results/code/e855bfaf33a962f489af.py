import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    n, m, p = 3, 5, 6
    
    # Helper to create a rank-1 term block
    def add_rank1_block(U, V, W, i_offset, j_offset, k_offset, rank_offset, u_val, v_val, w_val):
        # u_val is scalar for U[i_offset, rank_offset]
        # v_val is scalar for V[j_offset, rank_offset]
        # w_val is scalar for W[k_offset, rank_offset]
        U[i_offset][rank_offset] += u_val
        V[j_offset][rank_offset] += v_val
        W[k_offset][rank_offset] += w_val

    # Helper for Schoolbook 1x1x1
    def add_schoolbook_1x1x1(U, V, W, i, j, k, rank_offset):
        add_rank1_block(U, V, W, i, j, k, rank_offset, 1, 1, 1)

    # Helper for Strassen 2x2x2
    # Indices in U: 0,1,2,3 (i*m+j)
    # Indices in V: 0,1,2,3 (j*p+k)
    # Indices in W: 0,1,2,3 (k*n+i)
    def add_strassen_2x2x2(U, V, W, i_base, j_base, k_base, rank_base):
        # Map local indices to global
        # U local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        add_rank1_block(U, V, W, i_base + 0, j_base + 0, k_base + 0, rank_base + 0, 1, 1, 1)
        add_rank1_block(U, V, W, i_base + 3, j_base + 3, k_base + 0, rank_base + 0, 1, 1, 1)
        # Wait, M1 contributes to C00 and C11.
        # W00 (C00) gets 1. W11 (C11) gets 1.
        # My add_rank1_block adds to a single W entry.
        # Strassen terms contribute to multiple W entries.
        # I need a different helper.
        
        # Let's define the Strassen coefficients explicitly for U, V, W columns.
        # U col 0: [1, 0, 0, 1] -> A00, A11
        # V col 0: [1, 0, 0, 1] -> B00, B11
        # W col 0: [1, 0, 0, 1] -> C00, C11
        
        # M2 = (A10+A11)B00
        # U col 1: [0, 0, 1, 1] -> A10, A11
        # V col 1: [1, 0, 0, 0] -> B00
        # W col 1: [0, 0, 1, 0] -> C10
        
        # M3 = A00(B01-B11)
        # U col 2: [1, 0, 0, 0] -> A00
        # V col 2: [0, 1, 0, -1] -> B01, -B11
        # W col 2: [0, 1, 0, 0] -> C01
        
        # M4 = A11(B10-B00)
        # U col 3: [0, 0, 0, 1] -> A11
        # V col 3: [-1, 0, 1, 0] -> -B00, B10
        # W col 3: [1, 0, 1, 0] -> C00, C10
        
        # M5 = (A00+A01)B11
        # U col 4: [1, 1, 0, 0] -> A00, A01
        # V col 4: [0, 0, 0, 1] -> B11
        # W col 4: [-1, 0, 0, 1] -> -C00, C11
        
        # M6 = (A10-A00)(B00+B01)
        # U col 5: [-1, 0, 1, 0] -> -A00, A10
        # V col 5: [1, 1, 0, 0] -> B00, B01
        # W col 5: [0, 0, 0, 1] -> C11
        
        # M7 = (A01-A11)(B10+B11)
        # U col 6: [0, 1, 0, -1] -> A01, -A11
        # V col 6: [0, 0, 1, 1] -> B10, B11
        # W col 6: [1, 0, 0, 0] -> C00
        
        # Initialize columns
        U_cols = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        V_cols = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        W_cols = [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [-1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for c in range(7):
            r = rank_base + c
            for li in range(4):
                if U_cols[c][li] != 0:
                    gi = i_base + li
                    U[gi][r] += U_cols[c][li]
            for lj in range(4):
                if V_cols[c][lj] != 0:
                    gj = j_base + lj
                    V[gj][r] += V_cols[c][lj]
            for lk in range(4):
                if W_cols[c][lk] != 0:
                    gk = k_base + lk
                    W[gk][r] += W_cols[c][lk]
                    
    # Helper for 1xm xp (Rank m)
    def add_1xm_p(U, V, W, i_base, j_base, k_base, m_size, p_size, rank_base):
        # A is 1 x m. B is m x p.
        # C_k = sum_j A_j B_jk
        # Rank m.
        for j in range(m_size):
            r = rank_base + j
            U[i_base][r] = 1
            V[j_base + j][r] = 1
            for k in range(p_size):
                W[k_base + k][r] = 1
                
    # Helper for nx1 x p (Rank p)
    def add_nx1_p(U, V, W, i_base, j_base, k_base, n_size, p_size, rank_base):
        # A is n x 1. B is 1 x p.
        # C_i = A_i B_0
        # Rank p? No, Rank 1?
        # C_i = A_i B_0.
        # U_i,1 = 1, V_0,1 = 1, W_0,1 = 1?
        # M_1 = A_i B_0.
        # C_i = 1 * M_1.
        # So W_i,1 = 1.
        # Rank 1.
        r = rank_base
        for i in range(n_size):
            U[i_base + i][r] = 1
        V[j_base][r] = 1
        for k in range(p_size):
            W[k_base + k][r] = 1
            # Wait, C_i = A_i B_0.
            # M_1 = (sum_i U_i A_i) (sum_k W_k B_k)?
            # No, M_1 = (sum_i U_i A_i) * (sum_k W_k B_k) is wrong.
            # M_1 = (sum_i U_i A_i) * B_0?
            # If V_0=1, then sum_j V_j A_j = A_0? No, A is n x 1.
            # A_j is scalar.
            # sum_j V_j A_j = A_0 if V_0=1.
            # Then M_1 = A_0 * (sum_k W_k B_k).
            # This is not A_i B_0.
            
            # Correct decomposition for n x 1 x p:
            # C_i = A_i B_0.
            # We need sum_c U_ic (sum_j V_jc A_j) (sum_k W_kc B_k) = A_i B_0.
            # Let r=1.
            # U_i1 = 1? No, U_i1 must depend on i to pick A_i.
            # V_01 = 1.
            # W_01 = 1? No, W_k1 must pick B_0.
            # If W_01=1, W_k1=0 for k!=0.
            # Then M_1 = (sum_j V_j1 A_j) * B_0.
            # We need sum_j V_j1 A_j = A_i.
            # This requires V_i1 = 1, V_j1 = 0 for j!=i.
            # But V is shared for all i. V is m x r = 1 x r.
            # So V has only 1 row.
            # We cannot have V_i1 depend on i.
            # So rank 1 is impossible.
            # We need r = n * p?
            # No, rank(n x 1 x p) = n * p?
            # T_i,0,k = 1 if i matches, k matches?
            # C_i = A_i B_0.
            # Terms are A_i B_0.
            # These are n distinct terms.
            # Can we combine?
            # A_1 B_0 + A_2 B_0 = (A_1+A_2) B_0.
            # We can sum A's.
            # C_i = A_i B_0.
            # This is not a sum of combined terms for different i.
            # Each C_i depends only on A_i.
            # So we need n separate multiplications A_i * B_0.
            # Rank = n.
            # Let r=n.
            # U_i,c = delta_ic.
            # V_0,c = 1.
            # W_0,c = 1.
            # M_c = (sum_j V_jc A_j) * (sum_k W_kc B_k) = A_c * B_0.
            # C_i = sum_c U_ic M_c = 1 * A_i B_0.
            # Correct.
            pass

    # Corrected add_nx1_p
    def add_nx1_p_correct(U, V, W, i_base, j_base, k_base, n_size, p_size, rank_base):
        # Rank n_size
        for c in range(n_size):
            r = rank_base + c
            U[i_base + c][r] = 1
            V[j_base][r] = 1
            W[k_base][r] = 1
            # Note: W index is k. We only care about k=0 for B_0?
            # B is 1 x p. B_0 is the only element in row 0?
            # No, B is 1 x p. Elements are B_{0,0}, B_{0,1}...
            # C_i = A_i B_{0,k}? No, C is n x p.
            # C_{i,k} = A_i B_{0,k}.
            # My previous logic assumed B is 1 x 1.
            # If B is 1 x p, then C_{i,k} = A_i B_{0,k}.
            # This is outer product of A (n x 1) and B (1 x p).
            # Rank 1.
            # U_i,1 = 1. V_0,1 = 1. W_k,1 = 1.
            # M_1 = (sum_j V_j A_j) (sum_k' W_k' B_{k'})?
            # No, M_1 = (A_0) * (B_k)?
            # V is 1 x 1. V_0=1. sum_j V_j A_j = A_0.
            # W is p x 1. W_k=1. sum_k' W_k' B_{k'} = sum_k' B_{k'}.
            # This is not B_k.
            
            # Correct decomposition for n x 1 x p:
            # C_{i,k} = A_i B_k.
            # Rank 1?
            # U_i,1 = 1.
            # V_0,1 = 1.
            # W_k,1 = 1.
            # M_1 = A_0 * B_k? No.
            # M_1 = (sum_j V_j A_j) * (sum_{j'} W_{j'} B_{j'})?
            # The second sum is over the interface index.
            # For B, interface is row 0.
            # So sum_{j'} W_{j'} B_{j',k}.
            # If W_{0,1}=1, others 0.
            # Then sum is B_{0,k}.
            # First sum: sum_j V_j A_{i,j}.
            # A is n x 1. Interface is col 0.
            # V is 1 x 1. V_0=1.
            # Sum is A_{i,0}.
            # So M_1 = A_{i,0} B_{0,k}.
            # C_{i,k} = sum_c U_{i,c} M_c.
            # If r=1, U_{i,1}=1.
            # C_{i,k} = A_{i,0} B_{0,k}.
            # Correct. Rank 1.
            
        # So add_nx1_p uses Rank 1.
        r = rank_base
        for i in range(n_size):
            U[i_base + i][r] = 1
        V[j_base][r] = 1
        for k in range(p_size):
            W[k_base + k][r] = 1

    # Helper for nxm x 1 (Rank n*m)
    def add_nxm_1(U, V, W, i_base, j_base, k_base, n_size, m_size, rank_base):
        # C_i = sum_j A_ij B_j
        # Rank n*m
        for i in range(n_size):
            for j in range(m_size):
                r = rank_base + i * m_size + j
                U[i_base + i][r] = 1
                V[j_base + j][r] = 1
                W[k_base][r] = 1

    # Initialize U, V, W
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # Estimate max rank
    max_rank = 90
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Split n=3 into 2+1
    # Split m=5 into 4+1
    # Split p=6 into 4+2
    
    n1, n2 = 2, 1
    m1, m2 = 4, 1
    p1, p2 = 4, 2
    
    # Block 1: 2x4x4
    # Split m1=4 into 2+2, p1=4 into 2+2
    # 4 sub-blocks of 2x2x2
    for mi in range(2):
        for pi in range(2):
            i_base = 0
            j_base = mi * 2
            k_base = pi * 2
            add_strassen_2x2x2(U, V, W, i_base, j_base, k_base, current_rank)
            current_rank += 7
    
    # Block 2: 2x4x2
    # Split m1=4 into 2+2
    # 2 sub-blocks of 2x2x2
    for mi in range(2):
        i_base = 0
        j_base = mi * 2
        k_base = p1
        add_strassen_2x2x2(U, V, W, i_base, j_base, k_base, current_rank)
        current_rank += 7
        
    # Block 3: 2x1x4
    # Rank 1 (n x 1 x p)
    i_base = 0
    j_base = m1
    k_base = 0
    add_nx1_p_correct(U, V, W, i_base, j_base, k_base, n1, p1, current_rank)
    current_rank += 1
    
    # Block 4: 2x1x2
    # Rank 1
    i_base = 0
    j_base = m1
    k_base = p1
    add_nx1_p_correct(U, V, W, i_base, j_base, k_base, n1, p2, current_rank)
    current_rank += 1
    
    # Block 5: 1x4x4
    # Rank 4 (1 x m x p)
    i_base = n1
    j_base = 0
    k_base = 0
    add_1xm_p(U, V, W, i_base, j_base, k_base, m1, p1, current_rank)
    current_rank += m1
    
    # Block 6: 1x4x2
    # Rank 4
    i_base = n1
    j_base = 0
    k_base = p1
    add_1xm_p(U, V, W, i_base, j_base, k_base, m1, p2, current_rank)
    current_rank += m1
    
    # Block 7: 1x1x4
    # Rank 1
    i_base = n1
    j_base = m1
    k_base = 0
    add_nx1_p_correct(U, V, W, i_base, j_base, k_base, n2, p1, current_rank)
    current_rank += 1
    
    # Block 8: 1x1x2
    # Rank 1
    i_base = n1
    j_base = m1
    k_base = p1
    add_nx1_p_correct(U, V, W, i_base, j_base, k_base, n2, p2, current_rank)
    current_rank += 1
    
    # Trim to current_rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
