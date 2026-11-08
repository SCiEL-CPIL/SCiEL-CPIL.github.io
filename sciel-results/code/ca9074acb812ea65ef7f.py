import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_offset: row offset in U
    # b_offset: row offset in V
    # c_offset: col offset in W (using trace convention W[k*n+i])
    # c_row_offset: row offset in C
    def add_strassen_2x2(a_row_start, b_row_start, b_col_start, c_row_start, c_col_start, col_offset):
        # U indices for A (2x2):
        # (a_row_start, b_col_start) -> U[ a_row_start*m + b_col_start ]
        # (a_row_start, b_col_start+1) -> U[ a_row_start*m + b_col_start+1 ]
        # (a_row_start+1, b_col_start) -> U[ (a_row_start+1)*m + b_col_start ]
        # (a_row_start+1, b_col_start+1) -> U[ (a_row_start+1)*m + b_col_start+1 ]
        
        # V indices for B (2x2):
        # (b_row_start, b_col_start) -> V[ b_row_start*p + b_col_start ]
        # ...
        
        # W indices for C (2x2):
        # Trace convention: W[k*n + i] where k is col, i is row.
        # C[i, j] corresponds to W[j*n + i]
        
        # Strassen Formulas:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)B11      -> C21 += M2
        # M3 = A11(B12-B22)      -> C12 += M3
        # M4 = A22(B21-B11)      -> C21 += M4
        # M5 = (A11+A12)B22      -> C12 += M5
        # M6 = (A21-A11)(B11+B12)-> C22 += M6
        # M7 = (A12-A22)(B21+B22)-> C11 += M7
        
        # M1
        U[a_row_start*m + b_col_start, col_offset] += 1
        U[(a_row_start+1)*m + (b_col_start+1), col_offset] += 1
        V[b_row_start*p + b_col_start, col_offset] += 1
        V[(b_row_start+1)*p + (b_col_start+1), col_offset] += 1
        W[c_col_start*n + c_row_start, col_offset] += 1
        W[(c_col_start+1)*n + (c_row_start+1), col_offset] += 1
        
        # M2
        U[(a_row_start+1)*m + b_col_start, col_offset] += 1
        U[(a_row_start+1)*m + (b_col_start+1), col_offset] += 1
        V[b_row_start*p + b_col_start, col_offset] += 1
        W[c_col_start*n + (c_row_start+1), col_offset] += 1
        
        # M3
        U[a_row_start*m + b_col_start, col_offset] += 1
        V[b_row_start*p + (b_col_start+1), col_offset] += 1
        V[(b_row_start+1)*p + (b_col_start+1), col_offset] -= 1
        W[(c_col_start+1)*n + c_row_start, col_offset] += 1
        
        # M4
        U[(a_row_start+1)*m + (b_col_start+1), col_offset] += 1
        V[(b_row_start+1)*p + b_col_start, col_offset] += 1
        V[b_row_start*p + b_col_start, col_offset] -= 1
        W[c_col_start*n + (c_row_start+1), col_offset] += 1
        
        # M5
        U[a_row_start*m + b_col_start, col_offset] += 1
        U[a_row_start*m + (b_col_start+1), col_offset] += 1
        V[(b_row_start+1)*p + (b_col_start+1), col_offset] += 1
        W[(c_col_start+1)*n + c_row_start, col_offset] += 1
        
        # M6
        U[(a_row_start+1)*m + b_col_start, col_offset] += 1
        U[a_row_start*m + b_col_start, col_offset] -= 1
        V[b_row_start*p + b_col_start, col_offset] += 1
        V[b_row_start*p + (b_col_start+1), col_offset] += 1
        W[(c_col_start+1)*n + (c_row_start+1), col_offset] += 1
        
        # M7
        U[a_row_start*m + (b_col_start+1), col_offset] += 1
        U[(a_row_start+1)*m + (b_col_start+1), col_offset] -= 1
        V[(b_row_start+1)*p + b_col_start, col_offset] += 1
        V[(b_row_start+1)*p + (b_col_start+1), col_offset] += 1
        W[c_col_start*n + c_row_start, col_offset] += 1
        
        return col_offset + 7

    # Helper to add a 1x2 * 2x2 -> 1x2 multiplication (rank 4)
    # A is 1x2 (a0, a1)
    # B is 2x2 (b00, b01, b10, b11)
    # C is 1x2 (c0, c1)
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    def add_1x2_mul(a_row, b_row_start, b_col_start, c_row, c_col_start, col_offset):
        # a0: U[a_row*m + b_col_start]
        # a1: U[a_row*m + b_col_start+1]
        
        # b00: V[b_row_start*p + b_col_start]
        # b01: V[b_row_start*p + b_col_start+1]
        # b10: V[(b_row_start+1)*p + b_col_start]
        # b11: V[(b_row_start+1)*p + b_col_start+1]
        
        # c0: W[c_col_start*n + c_row]
        # c1: W[(c_col_start+1)*n + c_row]
        
        # M1 = a0 * b00 -> c0
        U[a_row*m + b_col_start, col_offset] = 1
        V[b_row_start*p + b_col_start, col_offset] = 1
        W[c_col_start*n + c_row, col_offset] = 1
        col_offset += 1
        
        # M2 = a1 * b10 -> c0
        U[a_row*m + (b_col_start+1), col_offset] = 1
        V[(b_row_start+1)*p + b_col_start, col_offset] = 1
        W[c_col_start*n + c_row, col_offset] = 1
        col_offset += 1
        
        # M3 = a0 * b01 -> c1
        U[a_row*m + b_col_start, col_offset] = 1
        V[b_row_start*p + (b_col_start+1), col_offset] = 1
        W[(c_col_start+1)*n + c_row, col_offset] = 1
        col_offset += 1
        
        # M4 = a1 * b11 -> c1
        U[a_row*m + (b_col_start+1), col_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+1), col_offset] = 1
        W[(c_col_start+1)*n + c_row, col_offset] = 1
        col_offset += 1
        
        return col_offset

    current_col = 0
    
    # 1. C_top_left (rows 0-1, cols 0-1)
    # A_tl (rows 0-1, cols 0-1) * B_11 (rows 0-1, cols 0-1)
    current_col = add_strassen_2x2(0, 0, 0, 0, 0, current_col)
    
    # A_tr (rows 0-1, cols 2-3) * B_21 (rows 2-3, cols 0-1)
    current_col = add_strassen_2x2(0, 2, 0, 0, 0, current_col)
    
    # 2. C_top_right (rows 0-1, cols 2-3)
    # A_tl (rows 0-1, cols 0-1) * B_12 (rows 0-1, cols 2-3)
    current_col = add_strassen_2x2(0, 0, 2, 0, 2, current_col)
    
    # A_tr (rows 0-1, cols 2-3) * B_22 (rows 2-3, cols 2-3)
    current_col = add_strassen_2x2(0, 2, 2, 0, 2, current_col)
    
    # 3. C_bottom_left (row 2, cols 0-1)
    # A_bl (row 2, cols 0-1) * B_11 (rows 0-1, cols 0-1)
    current_col = add_1x2_mul(2, 0, 0, 2, 0, current_col)
    
    # A_br (row 2, cols 2-3) * B_21 (rows 2-3, cols 0-1)
    current_col = add_1x2_mul(2, 2, 0, 2, 0, current_col)
    
    # 4. C_bottom_right (row 2, cols 2-3)
    # A_bl (row 2, cols 0-1) * B_12 (rows 0-1, cols 2-3)
    current_col = add_1x2_mul(2, 0, 2, 2, 2, current_col)
    
    # A_br (row 2, cols 2-3) * B_22 (rows 2-3, cols 2-3)
    current_col = add_1x2_mul(2, 2, 2, 2, 2, current_col)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
