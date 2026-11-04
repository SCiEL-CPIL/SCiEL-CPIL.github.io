import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using block decomposition.
    Partitions into 2x2 and 3x3 blocks.
    Uses Strassen (rank 7) for 2x2x2 blocks.
    Uses Schoolbook for other blocks.
    Total rank: 7 (2x2x2) + 12 (2x3x2) + 12 (2x2x3) + 18 (2x3x3) + 
                12 (3x2x2) + 18 (3x3x2) + 12 (3x2x3) + 27 (3x3x3) = 118.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build the U, V, W matrices column by column.
    # First, determine the total rank.
    # Block 1: 2x2x2 (Strassen) -> 7
    # Block 2: 2x3x2 (Schoolbook) -> 12
    # Block 3: 2x2x3 (Schoolbook) -> 12
    # Block 4: 2x3x3 (Schoolbook) -> 18
    # Block 5: 3x2x2 (Schoolbook) -> 12
    # Block 6: 3x3x2 (Schoolbook) -> 18
    # Block 7: 3x2x3 (Schoolbook) -> 12
    # Block 8: 3x3x3 (Schoolbook) -> 27
    
    rank_block_1 = 7
    rank_block_2 = 12
    rank_block_3 = 12
    rank_block_4 = 18
    rank_block_5 = 12
    rank_block_6 = 18
    rank_block_7 = 12
    rank_block_8 = 27
    
    total_rank = rank_block_1 + rank_block_2 + rank_block_3 + rank_block_4 + \
                 rank_block_5 + rank_block_6 + rank_block_7 + rank_block_8
    
    U = np.zeros((nm, total_rank), dtype=np.int64)
    V = np.zeros((mp, total_rank), dtype=np.int64)
    W = np.zeros((pn, total_rank), dtype=np.int64)
    
    current_col = 0
    
    # Helper to add schoolbook decomposition for a block
    def add_schoolbook(n_block, m_block, p_block, 
                       a_row_start, a_col_start,
                       b_row_start, b_col_start,
                       c_row_start, c_col_start):
        nonlocal current_col
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    # Global indices
                    u_idx = (a_row_start + i) * m + (a_col_start + j)
                    v_idx = (b_row_start + j) * p + (b_col_start + k)
                    w_idx = (c_col_start + k) * n + (c_row_start + i) # Trace convention W[k*n+i, r]
                    
                    U[u_idx, current_col] = 1
                    V[v_idx, current_col] = 1
                    W[w_idx, current_col] = 1
                    current_col += 1

    # Helper to add Strassen decomposition for 2x2x2 block
    def add_strassen(a_row_start, a_col_start,
                     b_row_start, b_col_start,
                     c_row_start, c_col_start):
        nonlocal current_col
        
        # Strassen terms for 2x2 matrices
        # A indices: 00, 01, 10, 11 -> global (r, c)
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00 -> C10
        # M3 = A00(B01-B11) -> C01
        # M4 = A11(B10-B00) -> C00, C10
        # M5 = (A00+A01)B11 -> C00, C01
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        
        # Define local indices for 2x2 block
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Term 1
        U[(a_row_start+0)*m + (a_col_start+0), current_col] = 1
        U[(a_row_start+1)*m + (a_col_start+1), current_col] = 1
        V[(b_row_start+0)*p + (b_col_start+0), current_col] = 1
        V[(b_row_start+1)*p + (b_col_start+1), current_col] = 1
        W[(c_col_start+0)*n + (c_row_start+0), current_col] = 1
        W[(c_col_start+1)*n + (c_row_start+1), current_col] = 1
        current_col += 1
        
        # Term 2
        U[(a_row_start+1)*m + (a_col_start+0), current_col] = 1
        U[(a_row_start+1)*m + (a_col_start+1), current_col] = 1
        V[(b_row_start+0)*p + (b_col_start+0), current_col] = 1
        W[(c_col_start+0)*n + (c_row_start+1), current_col] = 1
        current_col += 1
        
        # Term 3
        U[(a_row_start+0)*m + (a_col_start+0), current_col] = 1
        V[(b_row_start+0)*p + (b_col_start+1), current_col] = 1
        V[(b_row_start+1)*p + (b_col_start+1), current_col] = -1
        W[(c_col_start+1)*n + (c_row_start+0), current_col] = 1
        current_col += 1
        
        # Term 4
        U[(a_row_start+1)*m + (a_col_start+1), current_col] = 1
        V[(b_row_start+1)*p + (b_col_start+0), current_col] = 1
        V[(b_row_start+0)*p + (b_col_start+0), current_col] = -1
        W[(c_col_start+0)*n + (c_row_start+0), current_col] = 1
        W[(c_col_start+0)*n + (c_row_start+1), current_col] = 1
        current_col += 1
        
        # Term 5
        U[(a_row_start+0)*m + (a_col_start+0), current_col] = 1
        U[(a_row_start+0)*m + (a_col_start+1), current_col] = 1
        V[(b_row_start+1)*p + (b_col_start+1), current_col] = 1
        W[(c_col_start+0)*n + (c_row_start+0), current_col] = -1
        W[(c_col_start+1)*n + (c_row_start+0), current_col] = 1
        current_col += 1
        
        # Term 6
        U[(a_row_start+1)*m + (a_col_start+0), current_col] = 1
        U[(a_row_start+0)*m + (a_col_start+0), current_col] = -1
        V[(b_row_start+0)*p + (b_col_start+0), current_col] = 1
        V[(b_row_start+0)*p + (b_col_start+1), current_col] = 1
        W[(c_col_start+1)*n + (c_row_start+1), current_col] = 1
        current_col += 1
        
        # Term 7
        U[(a_row_start+0)*m + (a_col_start+1), current_col] = 1
        U[(a_row_start+1)*m + (a_col_start+1), current_col] = -1
        V[(b_row_start+1)*p + (b_col_start+0), current_col] = 1
        V[(b_row_start+1)*p + (b_col_start+1), current_col] = 1
        W[(c_col_start+0)*n + (c_row_start+0), current_col] = 1
        W[(c_col_start+1)*n + (c_row_start+1), current_col] = 1
        current_col += 1

    # Block 1: C22 += A22 B22 (2x2x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: C22 += A23 B32 (2x3x2)
    # A23: rows 0-1, cols 2-4. B32: rows 2-4, cols 0-1. C22: rows 0-1, cols 0-1.
    add_schoolbook(2, 3, 2, 0, 2, 2, 0, 0, 0)
    
    # Block 3: C23 += A22 B23 (2x2x3)
    # A22: rows 0-1, cols 0-1. B23: rows 0-1, cols 2-4. C23: rows 0-1, cols 2-4.
    add_schoolbook(2, 2, 3, 0, 0, 0, 2, 0, 2)
    
    # Block 4: C23 += A23 B33 (2x3x3)
    # A23: rows 0-1, cols 2-4. B33: rows 2-4, cols 2-4. C23: rows 0-1, cols 2-4.
    add_schoolbook(2, 3, 3, 0, 2, 2, 2, 0, 2)
    
    # Block 5: C32 += A32 B22 (3x2x2)
    # A32: rows 2-4, cols 0-1. B22: rows 0-1, cols 0-1. C32: rows 2-4, cols 0-1.
    add_schoolbook(3, 2, 2, 2, 0, 0, 0, 2, 0)
    
    # Block 6: C32 += A33 B32 (3x3x2)
    # A33: rows 2-4, cols 2-4. B32: rows 2-4, cols 0-1. C32: rows 2-4, cols 0-1.
    add_schoolbook(3, 3, 2, 2, 2, 2, 0, 2, 0)
    
    # Block 7: C33 += A32 B23 (3x2x3)
    # A32: rows 2-4, cols 0-1. B23: rows 0-1, cols 2-4. C33: rows 2-4, cols 2-4.
    add_schoolbook(3, 2, 3, 2, 0, 0, 2, 2, 2)
    
    # Block 8: C33 += A33 B33 (3x3x3)
    # A33: rows 2-4, cols 2-4. B33: rows 2-4, cols 2-4. C33: rows 2-4, cols 2-4.
    add_schoolbook(3, 3, 3, 2, 2, 2, 2, 2, 2)
    
    if time.time() > deadline - safety_margin:
        pass # Just return
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
