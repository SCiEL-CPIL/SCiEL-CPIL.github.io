import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix using block decomposition with Strassen
    for 2x2 blocks.
    
    Strategy:
    - Split A (3x4) into blocks: rows {0,1}|{2}, cols {0,1}|{2,3}
      A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    - Split B (4x5) into blocks: rows {0,1}|{2,3}, cols {0,1}|{2,3}|{4}
      B11 (2x2), B12 (2x2), B13 (2x1)
      B21 (2x2), B22 (2x2), B23 (2x1)
    - C (3x5) blocks:
      C11 = A11*B11 + A12*B21  (2x2, two 2x2 products)
      C12 = A11*B12 + A12*B22  (2x2, two 2x2 products)
      C13 = A11*B13 + A12*B23  (2x1, two 2x2*2x1 products)
      C21 = A21*B11 + A22*B21  (1x2, two 1x2*2x2 products)
      C22 = A21*B12 + A22*B22  (1x2, two 1x2*2x2 products)
      C23 = A21*B13 + A22*B23  (1x1, two 1x2*2x1 products)
    
    For each pair of 2x2 matrix multiplications, we can use Strassen
    to compute both results with shared intermediates where possible.
    However, since the A and B blocks differ for each C block, we
    need separate Strassen decompositions.
    
    Actually, let's be more careful. Each C block requires two 2x2
    (or smaller) matrix multiplications. We can use Strassen for each
    individual 2x2 multiplication.
    
    Rank budget:
    - C11: 2 Strassen 2x2 mults = 2*7 = 14
    - C12: 2 Strassen 2x2 mults = 2*7 = 14
    - C13: 2 (2x2)*(2x1) mults. A 2x2 times 2x1 is a 2x1 result.
           This is equivalent to computing 2 dot products of length 2.
           Rank = 2*2 = 4 (schoolbook) or we can use a rank-3 method?
           Actually, (2x2)*(2x1) = (2x1) result. Each output element
           is a dot product of length 2. Two independent dot products
           of length 2: rank = 2*2 = 4.
           But wait, we can view this as a bilinear map from R^4 x R^2 -> R^2.
           The tensor rank of this is 4 (it's just two independent 2-dim
           dot products).
           So 2 such products = 8.
    - C21: 2 (1x2)*(2x2) mults. Similar to C13 but transposed.
           (1x2)*(2x2) = (1x2) result. Each output element is a dot
           product of length 2. Two independent dot products: rank = 4.
           2 such products = 8.
    - C22: Same as C21 = 8
    - C23: 2 (1x2)*(2x1) mults. (1x2)*(2x1) = (1x1) result.
           Each is a single dot product of length 2: rank = 2.
           2 such products = 4.
    
    Total rank = 14 + 14 + 8 + 8 + 8 + 4 = 56
    
    This is better than schoolbook 60!
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
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
    # For C = A*B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Index mapping for 2x2 matrix:
    # A[i,j] -> index i*2+j in the 2x2 block
    # For the full A (3x4), block A11 occupies rows 0-1, cols 0-1
    #   A11[i,j] = A[i,j] for i in {0,1}, j in {0,1}
    #   In flattened A (row-major, 3x4): index = i*4 + j
    # For B (4x5), block B11 occupies rows 0-1, cols 0-1
    #   B11[i,j] = B[i,j] for i in {0,1}, j in {0,1}
    #   In flattened B (row-major, 4x5): index = i*5 + j
    # For C (3x5), block C11 occupies rows 0-1, cols 0-1
    #   C11[i,j] = C[i,j] for i in {0,1}, j in {0,1}
    #   In flattened W (p*n = 5*3, row-major with k first): index = j*3 + i
    #   Wait, W[k*n+i, r] per the convention. So W index = k*n + i where k is col, i is row.
    #   C[i,j] corresponds to W[j*3 + i]
    
    def strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, 
                     c_row_offset, c_col_offset):
        """
        Add Strassen terms for multiplying a 2x2 block of A by a 2x2 block of B
        and adding the result to a 2x2 block of C.
        
        a_row_offset, a_col_offset: top-left corner of A block in full A (3x4)
        b_row_offset, b_col_offset: top-left corner of B block in full B (4x5)
        c_row_offset, c_col_offset: top-left corner of C block in full C (3x5)
        
        A is 3x4, flattened row-major: A[i,j] at index i*4+j
        B is 4x5, flattened row-major: B[i,j] at index i*5+j
        C is 3x5, W convention: C[i,j] at W index j*3+i
        """
        # A block entries: A[a_row_offset+i, a_col_offset+j] for i,j in {0,1}
        # B block entries: B[b_row_offset+i, b_col_offset+j] for i,j in {0,1}
        # C block entries: C[c_row_offset+i, c_col_offset+j] for i,j in {0,1}
        
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
    # A is 2x2, B is 2x1, C is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank 4 (schoolbook)
    def schoolbook_2x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 2x2 A block by 2x1 B block, add to 2x1 C block.
        A block: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, col b_col_offset
        C block: rows c_row_offset..c_row_offset+1, col c_col_offset
        """
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        add_term({(a_row_offset+0)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        add_term({(a_row_offset+0)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        add_term({(a_row_offset+1)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
        add_term({(a_row_offset+1)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
    
    # Schoolbook for (1x2)*(2x2) -> (1x2)
    # A is 1x2, B is 2x2, C is 1x2
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    def schoolbook_1x2_times_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 1x2 A block by 2x2 B block, add to 1x2 C block.
        A block: row a_row_offset, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        C block: row c_row_offset, cols c_col_offset..c_col_offset+1
        """
        # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        # C[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
    
    # Schoolbook for (1x2)*(2x1) -> (1x1)
    # A is 1x2, B is 2x1, C is 1x1
    # C = A[0,0]*B[0] + A[0,1]*B[1]
    def schoolbook_1x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        """
        Multiply 1x2 A block by 2x1 B block, add to 1x1 C block.
        A block: row a_row_offset, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, col b_col_offset
        C block: row c_row_offset, col c_col_offset
        """
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
    
    # Now build all blocks:
    # A blocks:
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, cols 2-3 (2x2)
    #   A21: row 2, cols 0-1 (1x2)
    #   A22: row 2, cols 2-3 (1x2)
    # B blocks:
    #   B11: rows 0-1, cols 0-1 (2x2)
    #   B12: rows 0-1, cols 2-3 (2x2)
    #   B13: rows 0-1, col 4 (2x1)
    #   B21: rows 2-3, cols 0-1 (2x2)
    #   B22: rows 2-3, cols 2-3 (2x2)
    #   B23: rows 2-3, col 4 (2x1)
    # C blocks:
    #   C11: rows 0-1, cols 0-1 (2x2) = A11*B11 + A12*B21
    #   C12: rows 0-1, cols 2-3 (2x2) = A11*B12 + A12*B22
    #   C13: rows 0-1, col 4 (2x1) = A11*B13 + A12*B23
    #   C21: row 2, cols 0-1 (1x2) = A21*B11 + A22*B21
    #   C22: row 2, cols 2-3 (1x2) = A21*B12 + A22*B22
    #   C23: row 2, col 4 (1x1) = A21*B13 + A22*B23
    
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
    
    # Check: for each (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should be 1
    # if (i,j,k) is a valid triple (which it always is for the multiplication tensor)
    # Actually, the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U_arr[i*m+j, t] * V_arr[j*p+k, t] * W_arr[k*n+i, t]
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
