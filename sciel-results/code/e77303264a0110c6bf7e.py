import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Strategy: Block decomposition with Strassen for 2x2 blocks.
    Rank = 56 (vs schoolbook 60).
    """
    n, m, p = DIMENSIONS
    deadline = DEADLINE
    
    # Build the decomposition by blocks
    # U: shape (n*m, rank) = (12, rank)
    # V: shape (m*p, rank) = (20, rank)  
    # W: shape (p*n, rank) = (15, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a single rank-1 term."""
        U_rows.append(u_dict)
        V_rows.append(v_dict)
        W_rows.append(w_dict)
    
    # Strassen 2x2 decomposition
    def strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, 
                     c_row_offset, c_col_offset):
        """
        Add Strassen terms for multiplying a 2x2 block of A by a 2x2 block of B
        and adding the result to a 2x2 block of C.
        """
        A00_idx = (a_row_offset + 0) * 4 + (a_col_offset + 0)
        A01_idx = (a_row_offset + 0) * 4 + (a_col_offset + 1)
        A10_idx = (a_row_offset + 1) * 4 + (a_col_offset + 0)
        A11_idx = (a_row_offset + 1) * 4 + (a_col_offset + 1)
        
        B00_idx = (b_row_offset + 0) * 5 + (b_col_offset + 0)
        B01_idx = (b_row_offset + 0) * 5 + (b_col_offset + 1)
        B10_idx = (b_row_offset + 1) * 5 + (b_col_offset + 0)
        B11_idx = (b_row_offset + 1) * 5 + (b_col_offset + 1)
        
        C00_idx = (c_col_offset + 0) * 3 + (c_row_offset + 0)
        C01_idx = (c_col_offset + 1) * 3 + (c_row_offset + 0)
        C10_idx = (c_col_offset + 0) * 3 + (c_row_offset + 1)
        C11_idx = (c_col_offset + 1) * 3 + (c_row_offset + 1)
        
        # M1 = (A00+A11)(B00+B11)
        add_term({A00_idx: 1, A11_idx: 1}, {B00_idx: 1, B11_idx: 1}, {C00_idx: 1, C11_idx: 1})
        # M2 = (A10+A11)B00
        add_term({A10_idx: 1, A11_idx: 1}, {B00_idx: 1}, {C10_idx: 1, C11_idx: -1})
        # M3 = A00(B10-B11)
        add_term({A00_idx: 1}, {B10_idx: 1, B11_idx: -1}, {C01_idx: 1, C11_idx: 1})
        # M4 = A11(B01-B10)
        add_term({A11_idx: 1}, {B01_idx: 1, B10_idx: -1}, {C00_idx: 1, C10_idx: 1})
        # M5 = (A00+A01)B11
        add_term({A00_idx: 1, A01_idx: 1}, {B11_idx: 1}, {C00_idx: -1, C01_idx: 1})
        # M6 = (A10-A00)(B00+B10)
        add_term({A10_idx: 1, A00_idx: -1}, {B00_idx: 1, B10_idx: 1}, {C11_idx: 1})
        # M7 = (A01-A11)(B01+B11)
        add_term({A01_idx: 1, A11_idx: -1}, {B01_idx: 1, B11_idx: 1}, {C00_idx: 1})
    
    # Schoolbook for (2x2)*(2x1) -> (2x1)
    def schoolbook_2x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 2x2 A block by 2x1 B block, add to 2x1 C block.
        """
        add_term({(a_row_offset+0)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        add_term({(a_row_offset+0)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        add_term({(a_row_offset+1)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
        add_term({(a_row_offset+1)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
    
    # Schoolbook for (1x2)*(2x2) -> (1x2)
    def schoolbook_1x2_times_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 1x2 A block by 2x2 B block, add to 1x2 C block.
        """
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
    
    # Schoolbook for (1x2)*(2x1) -> (1x1)
    def schoolbook_1x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 1x2 A block by 2x1 B block, add to 1x1 C block.
        """
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
    
    # Build all blocks:
    # C11 = A11*B11 + A12*B21
    strassen_2x2(0, 0, 0, 0, 0, 0)  # A11 * B11 -> C11
    strassen_2x2(0, 2, 2, 0, 0, 0)  # A12 * B21 -> C11
    
    # C12 = A11*B12 + A12*B22
    strassen_2x2(0, 0, 0, 2, 0, 2)  # A11 * B12 -> C12
    strassen_2x2(0, 2, 2, 2, 0, 2)  # A12 * B22 -> C12
    
    # C13 = A11*B13 + A12*B23
    schoolbook_2x2_times_2x1(0, 0, 0, 4, 0, 4)  # A11 * B13 -> C13
    schoolbook_2x2_times_2x1(0, 2, 2, 4, 0, 4)  # A12 * B23 -> C13
    
    # C21 = A21*B11 + A22*B21
    schoolbook_1x2_times_2x2(2, 0, 0, 0, 2, 0)  # A21 * B11 -> C21
    schoolbook_1x2_times_2x2(2, 2, 2, 0, 2, 0)  # A22 * B21 -> C21
    
    # C22 = A21*B12 + A22*B22
    schoolbook_1x2_times_2x2(2, 0, 0, 2, 2, 2)  # A21 * B12 -> C22
    schoolbook_1x2_times_2x2(2, 2, 2, 2, 2, 2)  # A22 * B22 -> C22
    
    # C23 = A21*B13 + A22*B23
    schoolbook_1x2_times_2x1(2, 0, 0, 4, 2, 4)  # A21 * B13 -> C23
    schoolbook_1x2_times_2x1(2, 2, 2, 4, 2, 4)  # A22 * B23 -> C23
    
    # Convert sparse dicts to dense arrays
    rank = len(U_rows)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        for idx, val in U_rows[t].items():
            U[idx][t] = val
        for idx, val in V_rows[t].items():
            V[idx][t] = val
        for idx, val in W_rows[t].items():
            W[idx][t] = val
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
