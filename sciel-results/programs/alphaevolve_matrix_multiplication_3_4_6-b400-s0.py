"""Matrix multiplication 3,4,6 rank-minimal decomposition."""

def construct():
    n, m, p = 3, 4, 6
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for A[i,j] * B[j,k] -> C[i,k]
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
    # Note: i_base, j_base, k_base are local to the sub-problem or global?
    # In the previous code, they were global indices.
    # We will use global indices here as well for consistency.
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

    # Part 1: M(1,4,6) for row 0 of A
    # A[0, :] * B[:, :] -> C[0, :]
    # Schoolbook: 4 * 6 = 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)

    # Part 2: M(2,4,6) for rows 1,2 of A
    # Split k (columns of B) into 3 groups of 2
    for k_block in range(3):
        k_base = k_block * 2
        
        # For each k_block, we have a 2x4x2 sub-problem
        # Split j (columns of A / rows of B) into 2 groups of 2
        for j_block in range(2):
            j_base = j_block * 2
            
            # i_base is 1 (since we are handling rows 1 and 2)
            i_base = 1
            
            # Add Strassen for this 2x2x2 block
            add_strassen(i_base, j_base, k_base)

    return {"U": U, "V": V, "W": W}
