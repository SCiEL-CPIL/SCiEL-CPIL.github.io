"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split m=4 into two blocks of 2.
    # For each block (size 3x2x7):
    #   Split n=3 into n=1 and n=2.
    #   n=1, m=2, p=7: Rank 1.
    #   n=2, m=2, p=7: Split p=7 into three p=2 and one p=1.
    #     p=2 blocks: 3 * Rank(2x2x2) = 3 * 7 = 21.
    #     p=1 block: Rank(2x2x1) = 1.
    #   Total for 3x2x7: 1 + 21 + 1 = 23.
    # Total Rank = 2 * 23 = 46.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Strassen vectors for 2x2x2
    # u, v, w are lists of 4 elements (flattened 2x2 matrices)
    # Indexing: i*2 + j for u, j*2 + k for v, k*2 + i for w
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 1, -1, 0],# M6
        [0, 0, 1, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, -1, 0, 1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 1, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 1, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 1, 0, 0], # M2 -> C10
        [0, 1, -1, 0],# M3 -> C10, C00
        [0, 0, 0, -1],# M4 -> C11
        [0, 0, 1, 0], # M5 -> C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Verify Strassen vectors quickly?
    # Sum_c u[i,j,c] v[j,k,c] w[k,i,c] should be 1.
    
    for block_m in range(2):
        j_off = block_m * 2
        
        # Sub-block 1: n=1 (i=0), m=2 (j=j_off, j_off+1), p=7
        # Rank 1
        # U[0, j_off], U[0, j_off+1] = 1
        # V[j_off, k], V[j_off+1, k] = 1 for all k
        # W[k, 0] = 1 for all k
        add_term(0 * m + j_off, j_off * p + 0, 0 * n + 0, 1)
        # Wait, add_term adds one column. We need to set multiple entries in the same column.
        # My add_term increments col. I need a helper to set a whole column.
        
    # Re-implement with column setting
    col = 0
    def set_column(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] = val
        for v_idx in v_indices:
            V[v_idx][col] = val
        for w_idx in w_indices:
            W[w_idx][col] = val
        col += 1

    for block_m in range(2):
        j_off = block_m * 2
        
        # Sub-block 1: n=1, m=2, p=7
        # U: (0, j_off), (0, j_off+1)
        # V: (j_off, k), (j_off+1, k) for k in 0..6
        # W: (k, 0) for k in 0..6
        u_idx = [0 * m + j_off, 0 * m + (j_off + 1)]
        v_idx = []
        for k in range(p):
            v_idx.append(j_off * p + k)
            v_idx.append((j_off + 1) * p + k)
        w_idx = [k * n + 0 for k in range(p)]
        set_column(u_idx, v_idx, w_idx, 1)
        
        # Sub-block 2: n=2 (i=1, 2), m=2, p=7
        # Split p into 2, 2, 2, 1
        for block_p in range(3):
            k_off = block_p * 2
            # Strassen 2x2x2
            # Local i' in 0,1 -> Global i in 1, 2
            # Local j' in 0,1 -> Global j in j_off, j_off+1
            # Local k' in 0,1 -> Global k in k_off, k_off+1
            
            for c in range(7):
                u_vals = strassen_u[c]
                v_vals = strassen_v[c]
                w_vals = strassen_w[c]
                
                u_indices = []
                for i_prime in range(2):
                    for j_prime in range(2):
                        val = u_vals[i_prime * 2 + j_prime]
                        if val != 0:
                            global_i = 1 + i_prime
                            global_j = j_off + j_prime
                            u_indices.append((global_i * m + global_j, val))
                
                v_indices = []
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = v_vals[j_prime * 2 + k_prime]
                        if val != 0:
                            global_j = j_off + j_prime
                            global_k = k_off + k_prime
                            v_indices.append((global_j * p + global_k, val))
                
                w_indices = []
                for k_prime in range(2):
                    for i_prime in range(2):
                        val = w_vals[k_prime * 2 + i_prime]
                        if val != 0:
                            global_k = k_off + k_prime
                            global_i = 1 + i_prime
                            w_indices.append((global_k * n + global_i, val))
                
                # Set column with values
                for idx, val in u_indices:
                    U[idx][col] = val
                for idx, val in v_indices:
                    V[idx][col] = val
                for idx, val in w_indices:
                    W[idx][col] = val
                col += 1
        
        # p=1 block (k=6)
        # n=2, m=2, p=1
        # Rank 1
        # U: (1, j), (2, j) for j in j_off, j_off+1
        # V: (j, 6) for j in j_off, j_off+1
        # W: (6, 1), (6, 2)
        u_idx = []
        for j_prime in range(2):
            u_idx.append((1 * m + (j_off + j_prime), 1))
            u_idx.append((2 * m + (j_off + j_prime), 1))
            
        v_idx = []
        for j_prime in range(2):
            v_idx.append(((j_off + j_prime) * p + 6, 1))
            
        w_idx = [6 * n + 1, 6 * n + 2]
        
        for idx, val in u_idx:
            U[idx][col] = val
        for idx, val in v_idx:
            V[idx][col] = val
        for idx in w_idx:
            W[idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
