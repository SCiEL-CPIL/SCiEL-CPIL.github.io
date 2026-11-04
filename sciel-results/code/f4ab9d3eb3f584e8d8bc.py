"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank accumulator
    U = []
    V = []
    W = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # Add a single rank-1 term
        # u_idx, v_idx, w_idx are the row indices in U, V, W
        # We append to the global lists
        pass # Placeholder, we will build matrices directly

    # We will build U, V, W as lists of lists, then convert or return as is.
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Since we don't know the final rank in advance, we can build columns.
    # Each column is a rank-1 term.
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        """
        u_dict: {row_index: value}
        v_dict: {row_index: value}
        w_dict: {row_index: value}
        """
        cols_U.append(u_dict)
        cols_V.append(v_dict)
        cols_W.append(w_dict)

    # 1. Six 2x2x2 Strassen blocks
    # Blocks defined by I, J, K sets
    # I = {0, 1}
    # J sets: {0,1}, {2,3}
    # K sets: {0,1}, {2,3}, {4,5}
    
    # Strassen for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 - M6
    # C11 = M1 - M3 + M2 - M5
    
    # We need to map these to U, V, W indices.
    # For a block with i0, i1, j0, j1, k0, k1:
    # U indices: i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1
    # V indices: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
    # W indices: k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # M1
        add_rank1_term(
            {i0*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k0*n+i1: -1, k1*n+i0: 1, k1*n+i1: -1} # C00=1, C01=0, C10=0, C11=-1? 
            # Wait, W corresponds to output C_ik.
            # Term contributes to C_ik with coefficient W_{k*n+i}.
            # M1 contributes to C00 (1), C11 (-1)?
            # C00 = M1... -> coeff 1 at W_{k0*n+i0}
            # C11 = ... - M1 -> coeff -1 at W_{k1*n+i1}
            # C01, C10 are 0.
        )
        # Correct W for M1:
        # k0, i0 -> 1
        # k1, i1 -> -1
        # Others 0.
        # Let's re-evaluate W indices.
        # W row index is k*n + i.
        
        # Let's just explicitly define the 7 terms for the block.
        
        # M1 = (A_i0_j0 + A_i1_j1) * (B_j0_k0 + B_j1_k1)
        # Contributes to C_i0_k0 (+1), C_i1_k1 (-1)
        add_rank1_term(
            {i0*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k1*n+i1: -1}
        )
        
        # M2 = (A_i1_j0 + A_i1_j1) * B_j0_k0
        # Contributes to C_i1_k0 (+1), C_i0_k0 (0), C_i1_k1 (0), C_i0_k1 (0)
        # Wait, C10 = M2... -> coeff 1 at W_{k0*n+i1}
        add_rank1_term(
            {i1*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1},
            {k0*n+i1: 1}
        )
        
        # M3 = A_i0_j0 * (B_j0_k1 - B_j1_k1)
        # Contributes to C_i0_k1 (+1)
        add_rank1_term(
            {i0*m+j0: 1},
            {j0*p+k1: 1, j1*p+k1: -1},
            {k1*n+i0: 1}
        )
        
        # M4 = A_i1_j1 * (B_j1_k0 - B_j0_k0)
        # Contributes to C_i1_k0 (-1), C_i0_k0 (-1)
        # C00 = ... - M4 -> coeff -1 at W_{k0*n+i0}
        # C10 = ... + M4 -> coeff 1 at W_{k0*n+i1}
        add_rank1_term(
            {i1*m+j1: 1},
            {j1*p+k0: 1, j0*p+k0: -1},
            {k0*n+i0: -1, k0*n+i1: 1}
        )
        
        # M5 = (A_i0_j0 + A_i0_j1) * B_j1_k1
        # Contributes to C_i0_k1 (+1), C_i1_k1 (-1)
        # C01 = ... + M5 -> coeff 1 at W_{k1*n+i0}
        # C11 = ... - M5 -> coeff -1 at W_{k1*n+i1}
        add_rank1_term(
            {i0*m+j0: 1, i0*m+j1: 1},
            {j1*p+k1: 1},
            {k1*n+i0: 1, k1*n+i1: -1}
        )
        
        # M6 = (A_i1_j0 - A_i0_j0) * (B_j0_k0 + B_j0_k1)
        # Contributes to C_i0_k0 (-1), C_i0_k1 (1)
        # C00 = ... + M6 -> coeff 1? No, C00 = M1 - M4 + M6 - M7.
        # So M6 coeff is +1 at C00.
        # C10 = ... - M6 -> coeff -1 at C10.
        # Wait, C01 = M3 + M5 + M7. M6 not in C01.
        # C11 = M1 - M3 + M2 - M5. M6 not in C11.
        # So M6 contributes to C00 (+1) and C10 (-1).
        add_rank1_term(
            {i1*m+j0: 1, i0*m+j0: -1},
            {j0*p+k0: 1, j0*p+k1: 1},
            {k0*n+i0: 1, k0*n+i1: -1}
        )
        
        # M7 = (A_i0_j1 - A_i1_j1) * (B_j1_k0 + B_j1_k1)
        # Contributes to C00 (-1), C01 (+1)
        # C00 = ... - M7 -> coeff -1.
        # C01 = ... + M7 -> coeff +1.
        add_rank1_term(
            {i0*m+j1: 1, i1*m+j1: -1},
            {j1*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: -1, k1*n+i0: 1}
        )
        
    # Define the 6 blocks
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_strassen_block(0, 1, 0, 1, 4, 5)
    add_strassen_block(0, 1, 2, 3, 4, 5)
    
    # 2. One 2x1x7 block for i in {0,1}, j=4
    # Rank 1
    # U: i0*m+4, i1*m+4
    # V: 4*p+k for k=0..6
    # W: k*n+i0, k*n+i1
    i0, i1 = 0, 1
    j0 = 4
    u_dict = {i0*m+j0: 1, i1*m+j0: 1}
    v_dict = {j0*p+k: 1 for k in range(p)}
    w_dict = {k*n+i0: 1 for k in range(p)}
    w_dict.update({k*n+i1: 1 for k in range(p)})
    add_rank1_term(u_dict, v_dict, w_dict)
    
    # 3. One 2x4x1 block for i in {0,1}, j in {0..3}, k=6
    # Rank 4 (one term per j)
    i0, i1 = 0, 1
    k0 = 6
    for j in range(4):
        u_dict = {i0*m+j: 1, i1*m+j: 1}
        v_dict = {j*p+k0: 1}
        w_dict = {k0*n+i0: 1, k0*n+i1: 1}
        add_rank1_term(u_dict, v_dict, w_dict)
        
    # 4. One 1x5x7 block for i=2
    # Rank 5 (one term per j)
    i0 = 2
    for j in range(m):
        u_dict = {i0*m+j: 1}
        v_dict = {j*p+k: 1 for k in range(p)}
        w_dict = {k*n+i0: 1 for k in range(p)}
        add_rank1_term(u_dict, v_dict, w_dict)
        
    # Convert lists of dicts to matrices
    rank = len(cols_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in cols_U[r].items():
            U[idx][r] = val
        for idx, val in cols_V[r].items():
            V[idx][r] = val
        for idx, val in cols_W[r].items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
