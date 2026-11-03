import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    R = 28
    U = [[0.0]*R for _ in range(8)]
    V = [[0.0]*R for _ in range(16)]
    W = [[0.0]*R for _ in range(8)]
    
    # Define the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # Each term is a list of (index, coeff) tuples
    # U indices: 0-7 for A[0,0..3], A[1,0..3]
    # V indices: 0-15 for B[0,0..3], B[1,0..3], B[2,0..3], B[3,0..3]
    # W indices: 0-7 for C[0,0..3], C[1,0..3] i.e., W[k*2+i]
    
    def apply_block(a_row, a_col_off, b_row_off, b_col_off, c_col_off, base):
        # a_row: 0 or 1 (which row of A, since A is 2x4)
        # a_col_off: column offset in A (0 or 2)
        # b_row_off: row offset in B (0 or 2)
        # b_col_off: col offset in B (0 or 2)
        # c_col_off: col offset in C (0 or 2)
        # base: starting column index
        
        terms = [
            # term 0: M1 = (x00+x11)(y00+y11)
            ([(a_row*m+a_col_off, 1.0), ((a_row+1)*m+a_col_off+1, 1.0)],
             [(b_row_off*p+b_col_off, 1.0), ((b_row_off+1)*p+b_col_off+1, 1.0)],
             [(c_col_off*n+0, 1.0), ((c_col_off+1)*n+0, 1.0),
              (c_col_off*n+1, 0.0), ((c_col_off+1)*n+1, 1.0)]),
            # term 1: M2 = (x10+x11)y00
            ([((a_row+1)*m+a_col_off, 1.0), ((a_row+1)*m+a_col_off+1, 1.0)],
             [(b_row_off*p+b_col_off, 1.0)],
             [(c_col_off*n+1, 1.0), ((c_col_off+1)*n+1, -1.0)]),
            # term 2: M3 = x00(y10-y11)
            ([(a_row*m+a_col_off, 1.0)],
             [((b_row_off+1)*p+b_col_off, 1.0), ((b_row_off+1)*p+b_col_off+1, -1.0)],
             [((c_col_off+1)*n+0, 1.0), ((c_col_off+1)*n+1, 1.0)]),
            # term 3: M4 = x11(y00-y10)
            ([((a_row+1)*m+a_col_off+1, 1.0)],
             [(b_row_off*p+b_col_off, 1.0), ((b_row_off+1)*p+b_col_off, -1.0)],
             [(c_col_off*n+0, 1.0), ((c_col_off+1)*n+0, 1.0), ((c_col_off+1)*n+1, -1.0)]),
            # term 4: M5 = x00(y10+y11)
            ([(a_row*m+a_col_off, 1.0)],
             [((b_row_off+1)*p+b_col_off, 1.0), ((b_row_off+1)*p+b_col_off+1, 1.0)],
             [(c_col_off*n+0, -1.0)]),
            # term 5: M6 = (x01-x11)y00
            ([(a_row*m+a_col_off+1, 1.0), ((a_row+1)*m+a_col_off+1, -1.0)],
             [(b_row_off*p+b_col_off, 1.0)],
             [(c_col_off*n+1, 1.0), ((c_col_off+1)*n+1, 1.0)]),
            # term 6: M7 = (x10-x00)(y01+y10)
            ([((a_row+1)*m+a_col_off, 1.0), (a_row*m+a_col_off, -1.0)],
             [(b_row_off*p+b_col_off+1, 1.0), ((b_row_off+1)*p+b_col_off, 1.0)],
             [(c_col_off*n+0, 1.0)]),
        ]
        
        for t, (u_terms, v_terms, w_terms) in enumerate(terms):
            col = base + t
            for idx, coeff in u_terms:
                U[idx][col] += coeff
            for idx, coeff in v_terms:
                V[idx][col] += coeff
            for idx, coeff in w_terms:
                W[idx][col] += coeff
    
    # C = A*B where A is 2x4, B is 4x4, C is 2x4
    # Split: A = [A1 A2] where A1=A[:,0:2], A2=A[:,2:4]
    # B = [[B11 B12],[B21 B22]]
    # C = [C1 C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    
    # C1 from A1*B11
    apply_block(0, 0, 0, 0, 0, 0)
    # C1 from A2*B21
    apply_block(0, 2, 2, 0, 0, 7)
    # C2 from A1*B12
    apply_block(0, 0, 0, 2, 2, 14)
    # C2 from A2*B22
    apply_block(0, 2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}
