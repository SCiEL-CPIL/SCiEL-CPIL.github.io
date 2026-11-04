"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_222():
    """
    Return Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2.
    Returns (U, V, W) where:
      U has shape (4, 7)  [indices: i*2+j]
      V has shape (4, 7)  [indices: j*2+k]
      W has shape (4, 7)  [indices: k*2+i]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # rows: 00,01,10,11
    V = [[0]*7 for _ in range(4)]  # rows: 00,01,10,11
    W = [[0]*7 for _ in range(4)]  # rows: 00,01,10,11 (k*2+i)
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=+1, A11=+1
    U[0][0] = 1; U[3][0] = 1
    # V: B00=+1, B11=+1
    V[0][0] = 1; V[3][0] = 1
    # W: C00=+1, C11=+1
    W[0][0] = 1; W[3][0] = 1  # C00: k=0,i=0; C11: k=1,i=1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1  # C10: k=0,i=1; C11: k=1,i=1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01: k=1,i=0; C11: k=1,i=1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[3][3] = 1  # C00: k=0,i=0; C11: k=1,i=1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00: k=0,i=0; C01: k=1,i=0
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11: k=1,i=1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00: k=0,i=0
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """
    Return schoolbook decomposition for n_block x m_block x p_block.
    U: (n_block*m_block, n_block*m_block*p_block)
    V: (m_block*p_block, n_block*m_block*p_block)
    W: (p_block*n_block, n_block*m_block*p_block)
    """
    rank = n_block * m_block * p_block
    U = [[0]*rank for _ in range(n_block * m_block)]
    V = [[0]*rank for _ in range(m_block * p_block)]
    W = [[0]*rank for _ in range(p_block * n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][col] = 1
                V[j * p_block + k][col] = 1
                W[k * n_block + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of (U, V, W, row_offsets, col_offsets) into full decomposition.
    Each decomp has local indices, we map to global.
    decomps: list of (U_local, V_local, W_local, i_off, j_off, k_off, p_off)
    where i_off, j_off are offsets into A's row/col, k_off into B's col, p_off into B's row
    Actually let's think of it as:
    Each sub-problem multiplies A[i_off:i_off+ni, j_off:j_off+mi] by B[j_off:j_off+mi, k_off:k_off+pi]
    to get C[i_off:i_off+ni, k_off:k_off+pi]
    
    For the sub-problem:
      U_local row index: i_local * mi + j_local -> global: (i_off + i_local) * m + (j_off + j_local)
      V_local row index: j_local * pi + k_local -> global: (j_off + j_local) * p + (k_off + k_local)
      W_local row index: k_local * ni + i_local -> global: (k_off + k_local) * n + (i_off + i_local)
    """
    total_rank = sum(len(U[0]) for U, V, W, *_ in decomps)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_loc, V_loc, W_loc, ni, mi, pi, i_off, j_off, k_off in decomps:
        r = len(U_loc[0])
        for i_loc in range(ni):
            for j_loc in range(mi):
                u_row = i_loc * mi + j_loc
                u_global = (i_off + i_loc) * m + (j_off + j_loc)
                for c in range(r):
                    U_full[u_global][col_offset + c] = U_loc[u_row][c]
        for j_loc in range(mi):
            for k_loc in range(pi):
                v_row = j_loc * pi + k_loc
                v_global = (j_off + j_loc) * p + (k_off + k_loc)
                for c in range(r):
                    V_full[v_global][col_offset + c] = V_loc[v_row][c]
        for k_loc in range(pi):
            for i_loc in range(ni):
                w_row = k_loc * ni + i_loc
                w_global = (k_off + k_loc) * n + (i_off + i_loc)
                for c in range(r):
                    W_full[w_global][col_offset + c] = W_loc[w_row][c]
        col_offset += r
    
    return U_full, V_full, W_full

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140
    
    # Partition:
    # A is 4x5: rows 0-1 (2 rows), cols 0-1 (2 cols) for Strassen block
    #            rows 0-1, cols 2-4 (2x3) schoolbook
    #            rows 2-3, cols 0-4 (2x5) schoolbook
    # B is 5x5: rows 0-1 (2 rows), cols 0-1 (2 cols) for Strassen block
    #            rows 0-1, cols 2-4 (2x3) schoolbook
    #            rows 2-4, cols 0-4 (3x5) schoolbook
    
    # The Strassen block: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is a (2,2,2) problem, rank 7 with Strassen
    
    # All other blocks are schoolbook
    
    # Let's enumerate all blocks:
    # A row partitions: [0:2], [2:4]
    # A col partitions: [0:2], [2:5]
    # B row partitions: [0:2], [2:5]  (must match A col partitions for the middle index)
    # B col partitions: [0:2], [2:5]
    # C row = A row, C col = B col
    
    # For each combination of (A_row_part, A_col_part, B_col_part) where A_col_part == B_row_part:
    blocks = []
    
    # A row ranges: (0,2), (2,4)
    # A col / B row ranges: (0,2), (2,5)
    # B col ranges: (0,2), (2,5)
    
    a_row_ranges = [(0, 2), (2, 4)]
    a_col_b_row_ranges = [(0, 2), (2, 5)]
    b_col_ranges = [(0, 2), (2, 5)]
    
    decomps = []
    
    for (i0, i1) in a_row_ranges:
        ni = i1 - i0
        for (j0, j1) in a_col_b_row_ranges:
            mi = j1 - j0
            for (k0, k1) in b_col_ranges:
                pi = k1 - k0
                
                if ni == 2 and mi == 2 and pi == 2:
                    # Use Strassen
                    U_loc, V_loc, W_loc = _strassen_222()
                else:
                    U_loc, V_loc, W_loc = _schoolbook_block(ni, mi, pi)
                
                decomps.append((U_loc, V_loc, W_loc, ni, mi, pi, i0, j0, k0))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
