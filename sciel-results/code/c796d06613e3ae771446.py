import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_222(U, V, W, a_row_start, a_col_start, b_row_start, b_col_start, offset):
        # A is 2x2 at a_row_start, a_col_start
        # B is 2x2 at b_row_start, b_col_start
        # C is 2x2, output indices i in 0..1, k in 0..1
        # W index is k*n + i
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a_row_start*m + a_col_start, offset] += 1
        U[(a_row_start+1)*m + (a_col_start+1), offset] += 1
        V[b_row_start*p + b_col_start, offset] += 1
        V[(b_row_start+1)*p + (b_col_start+1), offset] += 1
        # C00 += M1, C11 += M1
        W[0*n + 0, offset] += 1
        W[1*n + 1, offset] += 1
        
        # M2 = (A10 + A11) * B00
        U[(a_row_start+1)*m + a_col_start, offset+1] += 1
        U[(a_row_start+1)*m + (a_col_start+1), offset+1] += 1
        V[b_row_start*p + b_col_start, offset+1] += 1
        # C10 += M2, C11 -= M2
        W[0*n + 1, offset+1] += 1
        W[1*n + 1, offset+1] -= 1
        
        # M3 = A00 * (B10 - B11)
        U[a_row_start*m + a_col_start, offset+2] += 1
        V[(b_row_start+1)*p + b_col_start, offset+2] += 1
        V[(b_row_start+1)*p + (b_col_start+1), offset+2] -= 1
        # C01 += M3, C11 += M3
        W[1*n + 0, offset+2] += 1
        W[1*n + 1, offset+2] += 1
        
        # M4 = A11 * (B01 - B00)
        U[(a_row_start+1)*m + (a_col_start+1), offset+3] += 1
        V[b_row_start*p + (b_col_start+1), offset+3] += 1
        V[b_row_start*p + b_col_start, offset+3] -= 1
        # C00 += M4, C10 += M4
        W[0*n + 0, offset+3] += 1
        W[0*n + 1, offset+3] += 1
        
        # M5 = (A00 + A01) * B11
        U[a_row_start*m + a_col_start, offset+4] += 1
        U[a_row_start*m + (a_col_start+1), offset+4] += 1
        V[(b_row_start+1)*p + (b_col_start+1), offset+4] += 1
        # C00 -= M5, C01 += M5
        W[0*n + 0, offset+4] -= 1
        W[1*n + 0, offset+4] += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(a_row_start+1)*m + a_col_start, offset+5] += 1
        U[a_row_start*m + a_col_start, offset+5] -= 1
        V[b_row_start*p + b_col_start, offset+5] += 1
        V[b_row_start*p + (b_col_start+1), offset+5] += 1
        # C11 += M6
        W[1*n + 1, offset+5] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a_row_start*m + (a_col_start+1), offset+6] += 1
        U[(a_row_start+1)*m + (a_col_start+1), offset+6] -= 1
        V[(b_row_start+1)*p + b_col_start, offset+6] += 1
        V[(b_row_start+1)*p + (b_col_start+1), offset+6] += 1
        # C00 += M7, C11 += M7
        W[0*n + 0, offset+6] += 1
        W[1*n + 1, offset+6] += 1

    def add_schoolbook_221(U, V, W, a_row_start, a_col_start, b_row_start, b_col_start, offset):
        # A is 2x2, B is 2x1
        # C is 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # M0 = A00 * B0
        U[a_row_start*m + a_col_start, offset] += 1
        V[b_row_start*p + b_col_start, offset] += 1
        W[0*n + 0, offset] += 1 # C0
        
        # M1 = A01 * B1
        U[a_row_start*m + (a_col_start+1), offset+1] += 1
        V[(b_row_start+1)*p + b_col_start, offset+1] += 1
        W[0*n + 0, offset+1] += 1 # C0
        
        # M2 = A10 * B0
        U[(a_row_start+1)*m + a_col_start, offset+2] += 1
        V[b_row_start*p + b_col_start, offset+2] += 1
        W[0*n + 1, offset+2] += 1 # C1
        
        # M3 = A11 * B1
        U[(a_row_start+1)*m + (a_col_start+1), offset+3] += 1
        V[(b_row_start+1)*p + b_col_start, offset+3] += 1
        W[0*n + 1, offset+3] += 1 # C1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
    # B[0:2, 0:2]
    add_strassen_222(U, V, W, 0, 0, 0, 0, 0)
    # B[0:2, 2:4]
    add_strassen_222(U, V, W, 0, 0, 0, 2, 7)
    # B[0:2, 4:5]
    add_schoolbook_221(U, V, W, 0, 0, 0, 4, 14)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:5]
    # B[2:4, 0:2]
    add_strassen_222(U, V, W, 0, 2, 2, 0, 18)
    # B[2:4, 2:4]
    add_strassen_222(U, V, W, 0, 2, 2, 2, 25)
    # B[2:4, 4:5]
    add_schoolbook_221(U, V, W, 0, 2, 2, 4, 32)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
