"""Matrix multiplication 3,4,6 rank-minimal decomposition."""

def construct():
    n, m, p = 3, 4, 6
    
    # We will construct a rank 68 decomposition.
    # Structure:
    # Split m=4 into two 2s.
    # For each m-block, split p=6 into two 3s.
    # For each p-block, split n=3 into 2 and 1.
    #   n=2 part: split p=3 into 2 and 1.
    #     p=2 part: 2x2x2 Strassen (rank 7)
    #     p=1 part: 2x2x1 Schoolbook (rank 4)
    #   n=1 part: 1x2x3 Schoolbook (rank 6)
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        # Note: A single "term" in the tensor decomposition is a rank-1 update.
        # However, Strassen combines multiple scalar multiplications into one rank-1 term 
        # by having multiple non-zeros in U, V, W.
        # So we don't increment col for each scalar, but for each rank-1 component.
        
    # Helper to add a schoolbook term for A[i,j] * B[j,k] -> C[i,k]
    # This is a single rank-1 component with 1 non-zero in each vector.
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    # i_base, j_base, k_base are the global starting indices for the 2x2 block
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        
        # Indices for A (2x2)
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        # Indices for B (2x2)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # Indices for C (2x2) -> W index is k*n + i
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Iterate through the blocks
    for m_block in range(2):
        j_base = m_block * 2
        for p_block in range(2):
            k_base = p_block * 3
            
            # Part 1: n=2 (rows 0, 1)
            i_base = 0
            # Sub-Part 1.1: 2x2x2 Strassen (k 0,1)
            add_strassen(i_base, j_base, k_base)
            
            # Sub-Part 1.2: 2x2x1 Schoolbook (k 2)
            for i in range(2):
                for j in range(2):
                    add_schoolbook(i_base + i, j_base + j, k_base + 2)
            
            # Part 2: n=1 (row 2)
            i_base = 2
            # Sub-Part 2.1: 1x2x3 Schoolbook
            for j in range(2):
                for k in range(3):
                    add_schoolbook(i_base, j_base + j, k_base + k)

    return {"U": U, "V": V, "W": W}
