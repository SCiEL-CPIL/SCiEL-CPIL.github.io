"""Matrix multiplication 5x5x5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_block(u_rows, v_rows, w_rows, base_r, A_block_size, B_block_size, C_block_size):
    """
    Create Strassen rank-7 tensors for 2x2x2 blocks.
    A_block_size: size of A sub-matrix (n_a x m_a)
    B_block_size: size of B sub-matrix (m_b x p_b)
    C_block_size: size of C sub-matrix (p_c x n_c)
    
    Returns list of (u_indices, v_indices, w_indices, coeffs) for each of 7 rank-1 terms.
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2x2 with indices (i,j,k) where i,j in {0,1}
    # A[i,j], B[j,k], C[i,k] -> output M[k,i]
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M2 + M3 + M7
    
    # u: (n_a*m_a, rank) - A part
    # v: (m_b*p_b, rank) - B part  
    # w: (p_c*n_c, rank) - C part (output M[k,i])
    
    terms = []
    
    # Define the 7 products
    # Each product: (A_indices, A_coeffs, B_indices, B_coeffs, C_output_positions, C_coeffs)
    # A indices are (i,j) pairs, B indices are (j,k) pairs, C positions are (k,i) pairs
    
    prod_defs = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M2 = (A21+A22)B11
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        # M3 = A11(B21-B11)
        {
            'A': [((0,0), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M4 = A22(B12-B22)
        {
            'A': [((1,1), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M5 = (A11+A12)B22
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,0), 1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
    ]
    
    for pd in prod_defs:
        u_entries = []
        v_entries = []
        w_entries = []
        
        for (i, j), coeff in pd['A']:
            u_entries.append((i, j, coeff))
        for (j, k), coeff in pd['B']:
            v_entries.append((j, k, coeff))
        for (k, i), coeff in pd['C']:
            w_entries.append((k, i, coeff))
        
        terms.append({'u': u_entries, 'v': v_entries, 'w': w_entries})
    
    return terms

def _build_block_decomposition(n, m, p, block_structure):
    """
    Build tensor decomposition using block structure.
    block_structure: list of tuples (n_a, m_a, m_b, p_b, p_c, n_c, use_strassen)
    where:
      n_a, m_a: size of A block (rows x cols)
      m_b, p_b: size of B block (rows x cols)
      p_c, n_c: size of C block (rows x cols)
      use_strassen: whether to use Strassen (only for 2x2x2 blocks)
    """
    U = [[0] * 0 for _ in range(n * m)]  # Will be resized
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    total_rank = 0
    U_data = []
    V_data = []
    W_data = []
    
    for (n_a, m_a, m_b, p_b, p_c, n_c, use_strassen) in block_structure:
        if use_strassen:
            if n_a != 2 or m_a != 2 or m_b != 2 or p_b != 2 or p_c != 2 or n_c != 2:
                # Fall back to schoolbook if not 2x2x2
                use_strassen = False
        
        if use_strassen:
            rank_here = 7
        else:
            rank_here = n_a * m_a * p_b
        
        # Initialize temporary matrices for this block
        for r in range(rank_here):
            u_row = [0] * (n * m)
            v_row = [0] * (m * p)
            w_row = [0] * (p * n)
            
            if use_strassen:
                terms = _strassen_block(None, None, None, 0, 2, 2, 2)
                for t_idx, term in enumerate(terms):
                    # u part
                    for (i, j, coeff) in term['u']:
                        # Map local (i,j) to global A index
                        # This is simplified - actual mapping depends on block position
                        # For now, we'll handle this in the main function
                        pass
                    # v part
                    for (j, k, coeff) in term['v']:
                        pass
                    # w part
                    for (k, i, coeff) in term['w']:
                        pass
            else:
                col = 0
                for i in range(n_a):
                    for j in range(m_a):
                        for k in range(p_b):
                            u_row[i * m_a + j] = 1
                            v_row[j * p_b + k] = 1
                            w_row[k * n_c + i] = 1
                            col += 1
            
            U_data.append(u_row)
            V_data.append(v_row)
            W_data.append(w_row)
    
    return U_data, V_data, W_data

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U shape: (n*m, rank) = (25, rank)
    V shape: (m*p, rank) = (25, rank)
    W shape: (p*n, rank) = (25, rank)
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Use 2+2+1 block partition:
    # A is 5x5, split as rows 0-1, 2-3, 4 and cols 0-1, 2-3, 4
    # B is 5x5, split as rows 0-1, 2-3, 4 and cols 0-1, 2-3, 4
    # C is 5x5, split as rows 0-1, 2-3, 4 and cols 0-1, 2-3, 4
    
    # Block positions (n_a, m_a, m_b, p_b, p_c, n_c, use_strassen, A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    blocks = [
        # 2x2x2 blocks (use Strassen)
        (2, 2, 2, 2, 2, 2, True, 0, 0, 0, 0, 0, 0),    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (2, 2, 2, 2, 2, 2, True, 0, 0, 0, 2, 0, 2),    # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (2, 2, 2, 2, 2, 2, True, 0, 2, 2, 0, 0, 0),    # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (2, 2, 2, 2, 2, 2, True, 0, 2, 2, 2, 0, 2),    # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        (2, 2, 2, 2, 2, 2, True, 2, 0, 0, 0, 2, 0),    # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
        (2, 2, 2, 2, 2, 2, True, 2, 0, 0, 2, 2, 2),    # A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
        (2, 2, 2, 2, 2, 2, True, 2, 2, 2, 0, 2, 0),    # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        (2, 2, 2, 2, 2, 2, True, 2, 2, 2, 2, 2, 2),    # A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        # 1x1x1 blocks (schoolbook)
        (1, 1, 1, 1, 1, 1, False, 4, 4, 4, 4, 4, 4),   # A[4,4] * B[4,4] -> C[4,4]
        # Cross blocks (mixed sizes)
        (2, 1, 1, 2, 1, 2, False, 0, 4, 4, 0, 0, 0),   # A[0:2,4] * B[4,0:2] -> C[0:2,0:2]
        (2, 1, 1, 2, 1, 2, False, 0, 4, 4, 2, 0, 2),   # A[0:2,4] * B[4,2:4] -> C[0:2,2:4]
        (1, 2, 2, 1, 2, 1, False, 4, 0, 0, 4, 4, 0),   # A[4,0:2] * B[0:2,4] -> C[4,0]
        (1, 2, 2, 1, 2, 1, False, 4, 0, 0, 2, 4, 2),   # A[4,0:2] * B[0:2,4] -> C[4,2]
        (2, 1, 1, 2, 1, 2, False, 2, 4, 4, 0, 2, 0),   # A[2:4,4] * B[4,0:2] -> C[2:4,0:2]
        (2, 1, 1, 2, 1, 2, False, 2, 4, 4, 2, 2, 2),   # A[2:4,4] * B[4,2:4] -> C[2:4,2:4]
        (1, 2, 2, 1, 2, 1, False, 4, 2, 2, 4, 4, 0),   # A[4,2:4] * B[2:4,4] -> C[4,0]
        (1, 2, 2, 1, 2, 1, False, 4, 2, 2, 4, 4, 2),   # A[4,2:4] * B[2:4,4] -> C[4,2]
        (1, 1, 1, 2, 1, 1, False, 4, 4, 0, 4, 4, 0),   # A[4,4] * B[0,4] -> C[4,0]
        (1, 1, 1, 2, 1, 1, False, 4, 4, 0, 2, 4, 2),   # A[4,4] * B[0,4] -> C[4,2]
        (1, 1, 2, 1, 1, 1, False, 4, 4, 2, 4, 4, 0),   # A[4,4] * B[2,4] -> C[4,0]
        (1, 1, 2, 1, 1, 1, False, 4, 4, 2, 2, 4, 2),   # A[4,4] * B[2,4] -> C[4,2]
    ]
    
    # Collect all terms
    all_u = []
    all_v = []
    all_w = []
    
    for (n_a, m_a, m_b, p_b, p_c, n_c, use_strassen, 
         a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off) in blocks:
        
        if use_strassen:
            rank_here = 7
        else:
            rank_here = n_a * m_a * p_b
        
        for r in range(rank_here):
            u_row = [0] * (n * m)
            v_row = [0] * (m * p)
            w_row = [0] * (p * n)
            
            if use_strassen:
                terms = _strassen_block(None, None, None, 0, 2, 2, 2)
                for t_idx, term in enumerate(terms):
                    # u part
                    for (i, j, coeff) in term['u']:
                        global_i = a_row_off + i
                        global_j = a_col_off + j
                        u_row[global_i * m + global_j] += coeff
                    # v part
                    for (j, k, coeff) in term['v']:
                        global_j = b_row_off + j
                        global_k = b_col_off + k
                        v_row[global_j * p + global_k] += coeff
                    # w part
                    for (k, i, coeff) in term['w']:
                        global_k = c_row_off + k
                        global_i = c_col_off + i
                        w_row[global_k * n + global_i] += coeff
            else:
                col = 0
                for i in range(n_a):
                    for j in range(m_a):
                        for k in range(p_b):
                            global_i = a_row_off + i
                            global_j = a_col_off + j
                            global_j_b = b_row_off + j
                            global_k = b_col_off + k
                            global_k_c = c_row_off + k
                            global_i_c = c_col_off + i
                            u_row[global_i * m + global_j] = 1
                            v_row[global_j_b * p + global_k] = 1
                            w_row[global_k_c * n + global_i_c] = 1
                            col += 1
            
            all_u.append(u_row)
            all_v.append(v_row)
            all_w.append(w_row)
        
        if time.time() > deadline - 10:
            # Emergency fallback
            break
    
    rank = len(all_u)
    
    # Transpose to get (n*m, rank) format
    U = [[all_u[r][c] for r in range(rank)] for c in range(n * m)]
    V = [[all_v[r][c] for r in range(rank)] for c in range(m * p)]
    W = [[all_w[r][c] for r in range(rank)] for c in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
