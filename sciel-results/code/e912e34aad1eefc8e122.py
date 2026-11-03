"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # We'll build the decomposition block by block
    # Partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4}
    
    U_cols = []  # list of columns for U (each column is a vector of length n*m = 10)
    V_cols = []  # list of columns for V (each column is a vector of length m*p = 25)
    W_cols = []  # list of columns for W (each column is a vector of length p*n = 10)
    
    def add_term(u_val, v_val, w_val):
        """Add one rank-1 term. u_val, v_val, w_val are dicts mapping (index) to coefficient."""
        # Convert to full vectors
        U_vec = [0] * (n * m)
        V_vec = [0] * (m * p)
        W_vec = [0] * (p * n)
        
        for idx, coeff in u_val.items():
            U_vec[idx] += coeff
        for idx, coeff in v_val.items():
            V_vec[idx] += coeff
        for idx, coeff in w_val.items():
            W_vec[idx] += coeff
        
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
    
    # Strassen 2x2x2 decomposition
    # A is 2x2: A[i][j], B is 2x2: B[j][k], C is 2x2: C[i][k]
    # U index: i*m + j = i*5 + j (but within the block, we use local indices)
    # Actually, we need to map local block indices to global indices
    
    # Let me define helper: for a block at j_start, k_start, with local sizes
    # A local (i, j_local) -> global U index: i * m + (j_start + j_local)
    # B local (j_local, k_local) -> global V index: (j_start + j_local) * p + (k_start + k_local)
    # C local (i, k_local) -> global W index: (k_start + k_local) * n + i
    
    def strassen_2x2x2(j_start, k_start):
        """Add Strassen decomposition for 2x2x2 block at (j_start, k_start)."""
        # Local indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Global: j = j_start + j_local, k = k_start + k_local
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[0][0] -> U[0*5 + j_start + 0]
        # A22 = A[1][1] -> U[1*5 + j_start + 1]
        # B11 = B[0][0] -> V[(j_start+0)*5 + k_start + 0]
        # B22 = B[1][1] -> V[(j_start+1)*5 + k_start + 1]
        # C11 = C[0][0] -> W[(k_start+0)*2 + 0]
        # C22 = C[1][1] -> W[(k_start+1)*2 + 1]
        
        # M1: U has A[0][j_start+0] and A[1][j_start+1]
        #      V has B[j_start+0][k_start+0] and B[j_start+1][k_start+1]
        #      W has C[0][k_start+0] and C[1][k_start+1]
        
        # Let me define the 7 Strassen terms
        # Each term: u_dict, v_dict, w_dict
        
        # M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
        # C[0][0] += M1, C[1][1] += M1
        add_term(
            {0 * m + j_start + 0: 1, 1 * m + j_start + 1: 1},
            {(j_start + 0) * p + k_start + 0: 1, (j_start + 1) * p + k_start + 1: 1},
            {(k_start + 0) * n + 0: 1, (k_start + 1) * n + 1: 1}
        )
        
        # M2 = (A[1][0] + A[1][1]) * B[0][0]
        # C[1][0] += M2, C[1][1] -= M2
        add_term(
            {1 * m + j_start + 0: 1, 1 * m + j_start + 1: 1},
            {(j_start + 0) * p + k_start + 0: 1},
            {(k_start + 0) * n + 1: 1, (k_start + 1) * n + 1: -1}
        )
        
        # M3 = A[0][0] * (B[1][0] - B[1][1])
        # C[0][1] += M3, C[1][1] += M3
        add_term(
            {0 * m + j_start + 0: 1},
            {(j_start + 1) * p + k_start + 0: 1, (j_start + 1) * p + k_start + 1: -1},
            {(k_start + 1) * n + 0: 1, (k_start + 1) * n + 1: 1}
        )
        
        # M4 = A[1][1] * (B[0][1] - B[0][0])
        # C[0][0] += M4, C[1][0] += M4
        add_term(
            {1 * m + j_start + 1: 1},
            {(j_start + 0) * p + k_start + 1: 1, (j_start + 0) * p + k_start + 0: -1},
            {(k_start + 0) * n + 0: 1, (k_start + 0) * n + 1: 1}
        )
        
        # M5 = (A[0][0] + A[0][1]) * B[1][1]
        # C[0][0] -= M5, C[0][1] += M5
        add_term(
            {0 * m + j_start + 0: 1, 0 * m + j_start + 1: 1},
            {(j_start + 1) * p + k_start + 1: 1},
            {(k_start + 0) * n + 0: -1, (k_start + 1) * n + 0: 1}
        )
        
        # M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
        # C[1][1] += M6
        add_term(
            {1 * m + j_start + 0: 1, 0 * m + j_start + 0: -1},
            {(j_start + 0) * p + k_start + 0: 1, (j_start + 0) * p + k_start + 1: 1},
            {(k_start + 1) * n + 1: 1}
        )
        
        # M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
        # C[0][0] += M7
        add_term(
            {0 * m + j_start + 1: 1, 1 * m + j_start + 1: -1},
            {(j_start + 1) * p + k_start + 0: 1, (j_start + 1) * p + k_start + 1: 1},
            {(k_start + 0) * n + 0: 1}
        )
    
    # 2x2x1 block: A is 2x2, B is 2x1, C is 2x1
    # C[i][0] = A[i][0]*B[0][0] + A[i][1]*B[1][0]
    # Rank 4: just do it directly
    def block_2x2x1(j_start, k_start):
        for i in range(2):
            for j_local in range(2):
                add_term(
                    {i * m + j_start + j_local: 1},
                    {(j_start + j_local) * p + k_start + 0: 1},
                    {(k_start + 0) * n + i: 1}
                )
    
    # 2x1x2 block: A is 2x1, B is 1x2, C is 2x2
    # C[i][k] = A[i][0]*B[0][k]
    # Rank 4
    def block_2x1x2(j_start, k_start):
        for i in range(2):
            for k_local in range(2):
                add_term(
                    {i * m + j_start + 0: 1},
                    {(j_start + 0) * p + k_start + k_local: 1},
                    {(k_start + k_local) * n + i: 1}
                )
    
    # 2x1x1 block: A is 2x1, B is 1x1, C is 2x1
    # C[i][0] = A[i][0]*B[0][0]
    # Rank 2
    def block_2x1x1(j_start, k_start):
        for i in range(2):
            add_term(
                {i * m + j_start + 0: 1},
                {(j_start + 0) * p + k_start + 0: 1},
                {(k_start + 0) * n + i: 1}
            )
    
    # Now assemble the blocks
    # j blocks: {0,1} at j_start=0, {2,3} at j_start=2, {4} at j_start=4
    # k blocks: {0,1} at k_start=0, {2,3} at k_start=2, {4} at k_start=4
    
    j_starts = [0, 2, 4]
    k_starts = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_sizes = [2, 2, 1]
    
    for ji, j_start in enumerate(j_starts):
        for ki, k_start in enumerate(k_starts):
            js = j_sizes[ji]
            ks = k_sizes[ki]
            
            if js == 2 and ks == 2:
                strassen_2x2x2(j_start, k_start)
            elif js == 2 and ks == 1:
                block_2x2x1(j_start, k_start)
            elif js == 1 and ks == 2:
                block_2x1x2(j_start, k_start)
            elif js == 1 and ks == 1:
                block_2x1x1(j_start, k_start)
    
    # Convert lists to matrices
    U = [U_cols[c] for c in range(len(U_cols))]
    # Wait, U_cols is a list of vectors, each vector is a column
    # U should be (n*m) x rank, so U[i][c] = U_cols[c][i]
    rank = len(U_cols)
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n * m)]
    V = [[V_cols[c][i] for c in range(rank)] for i in range(m * p)]
    W = [[W_cols[c][i] for c in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
