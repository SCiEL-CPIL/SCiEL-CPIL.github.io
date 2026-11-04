import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        # This is not quite right, we need to add to the column for the current term
        # It's better to pass the column index and the local vectors
        
    # We will define the local decompositions as lists of (u_vec, v_vec, w_vec)
    # u_vec is length n_local, v_vec is length m_local, w_vec is length p_local
    # Wait, W is p x n. So w_vec is length p? No.
    # T_{i,j,k}. U is n, V is m, W is p?
    # Task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indices:
    # U row: i*m + j? No.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, c]"? The prompt says "U[i*m+j, c]" in the schoolbook comment.
    # Let's check the schoolbook code.
    # U[i * m + j][column] = 1. So U row index is i*m + j.
    # V[j * p + k][column] = 1. So V row index is j*p + k.
    # W[k * n + i][column] = 1. So W row index is k*n + i.
    
    # So for a local block with dimensions n_l, m_l, p_l at offsets (i0, j0, k0):
    # U row: (i0 + i_l) * m + (j0 + j_l)
    # V row: (j0 + j_l) * p + (k0 + k_l)
    # W row: (k0 + k_l) * n + (i0 + i_l)
    
    def add_decomposition(i0, j0, k0, n_l, m_l, p_l, decomp):
        nonlocal col
        for u_vec, v_vec, w_vec in decomp:
            # u_vec is length n_l? No, U is indexed by i, j.
            # In the local block, the "input" is A (n_l x m_l) and B (m_l x p_l).
            # The tensor is T_{i,j,k}.
            # U corresponds to A indices (i, j).
            # V corresponds to B indices (j, k).
            # W corresponds to C indices (k, i).
            
            # Wait, the standard tensor decomposition T = sum U_r V_r W_r
            # T_{i,j,k} = sum U_{i,r} V_{j,r} W_{k,r}?
            # If so, U is n x rank, V is m x rank, W is p x rank.
            # But the task requires U to be n*m x rank.
            # This implies the decomposition is for the bilinear map, not the trilinear form directly?
            # Or U_{i*m+j, r} is the coefficient of A_{i,j} in the r-th multiplication.
            # Yes.
            # Multiplication r: M_r = (sum_{i,j} U_{i*m+j, r} A_{i,j}) * (sum_{j,k} V_{j*p+k, r} B_{j,k})
            # Output C_{k,i} += W_{k*n+i, r} M_r.
            
            # So for a local block:
            # u_vec should be a vector of length n_l * m_l?
            # No, in Strassen, we defined U as length 4 (2x2).
            # So u_vec is length n_l * m_l.
            # v_vec is length m_l * p_l.
            # w_vec is length p_l * n_l.
            
            for idx_u, val_u in enumerate(u_vec):
                if val_u != 0:
                    i_l, j_l = divmod(idx_u, m_l)
                    global_u_idx = (i0 + i_l) * m + (j0 + j_l)
                    U[global_u_idx, col] += val_u
            
            for idx_v, val_v in enumerate(v_vec):
                if val_v != 0:
                    j_l, k_l = divmod(idx_v, p_l)
                    global_v_idx = (j0 + j_l) * p + (k0 + k_l)
                    V[global_v_idx, col] += val_v
            
            for idx_w, val_w in enumerate(w_vec):
                if val_w != 0:
                    k_l, i_l = divmod(idx_w, n_l)
                    global_w_idx = (k0 + k_l) * n + (i0 + i_l)
                    W[global_w_idx, col] += val_w
            
            col += 1

    # Define Strassen 2x2x2
    # u_vec length 4, v_vec length 4, w_vec length 4
    strassen_terms = []
    # M1
    strassen_terms.append(([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1])) # C00, C11?
    # Let's re-verify W mapping.
    # W index is k*n + i.
    # C00 is i=0, k=0 -> idx 0.
    # C01 is i=0, k=1 -> idx 1.
    # C10 is i=1, k=0 -> idx 2.
    # C11 is i=1, k=1 -> idx 3.
    
    # M1 contributes to C00 (1), C11 (1).
    # M2 contributes to C10 (1).
    # M3 contributes to C01 (1), C11 (1).
    # M4 contributes to C00 (1), C10 (1).
    # M5 contributes to C00 (-1), C01 (1).
    # M6 contributes to C11 (1).
    # M7 contributes to C00 (1).
    
    strassen_terms = [
        ([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [0, 0, 1, 1], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 1, 1, 0], [1, 0, 0, 0])  # M7
    ]
    
    # Define 1x2x4 Schoolbook
    # u_vec length 2 (1x2), v_vec length 8 (2x4), w_vec length 4 (4x1)
    schoolbook_124 = []
    for j in range(2):
        for k in range(4):
            u = [0, 0]
            u[j] = 1
            v = [0]*8
            v[j*4 + k] = 1
            w = [0]*4
            w[k] = 1
            schoolbook_124.append((u, v, w))
            
    # Now build the tree
    # 3x4x8
    # Split m=4 into 2+2
    # Block 1: i=0..2, j=0..1, k=0..7 (3x2x8)
    # Block 2: i=0..2, j=2..3, k=0..7 (3x2x8)
    
    for j_offset in [0, 2]:
        # 3x2x8 at (0, j_offset, 0)
        # Split p=8 into 4+4
        for k_offset in [0, 4]:
            # 3x2x4 at (0, j_offset, k_offset)
            # Split n=3 into 2+1
            # Top: 2x2x4 at (0, j_offset, k_offset)
            add_decomposition(0, j_offset, k_offset, 2, 2, 4, None) # Placeholder
            
            # Wait, I need to decompose 2x2x4 first.
            
    # Let's define a function to decompose 2x2x4
    def decompose_224(i0, j0, k0):
        # Split p=4 into 2+2
        for k_off in [0, 2]:
            # 2x2x2 at (i0, j0, k0+k_off)
            add_decomposition(i0, j0, k0+k_off, 2, 2, 2, strassen_terms)
            
    def decompose_124(i0, j0, k0):
        add_decomposition(i0, j0, k0, 1, 2, 4, schoolbook_124)
        
    def decompose_324(i0, j0, k0):
        # Top 2x2x4
        decompose_224(i0, j0, k0)
        # Bot 1x2x4
        decompose_124(i0+2, j0, k0)
        
    def decompose_328(i0, j0, k0):
        for k_off in [0, 4]:
            decompose_324(i0, j0, k0+k_off)
            
    def decompose_348():
        for j_off in [0, 2]:
            decompose_328(0, j_off, 0)
            
    decompose_348()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
