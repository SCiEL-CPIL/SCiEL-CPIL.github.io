import numpy as np

def construct():
    n, m, p = 3, 4, 6
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        for idx, coeff in zip(u_indices, u_coeffs):
            u_vec[idx] = coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            v_vec[idx] = coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            w_vec[idx] = coeff
            
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper for Strassen 2x2 block
    # A is at rows r0, r1 and cols c0, c1
    # B is at rows c0, c1 and cols d0, d1
    # C is at rows r0, r1 and cols d0, d1
    def add_strassen(r0, r1, c0, c1, d0, d1):
        # A indices
        a11 = r0*m + c0
        a12 = r0*m + c1
        a21 = r1*m + c0
        a22 = r1*m + c1
        
        # B indices
        b11 = c0*p + d0
        b12 = c0*p + d1
        b21 = c1*p + d0
        b22 = c1*p + d1
        
        # C indices (W[k*n+i, r] -> W[col*n + row])
        # C row r0, col d0 -> W[d0*n + r0]
        c11 = d0*n + r0
        c12 = d1*n + r0
        c21 = d0*n + r1
        c22 = d1*n + r1
        
        # M1 = (A11+A22)(B11+B22)
        # C11 += M1, C22 += M1
        add_term([a11, a22], [1, 1], [b11, b22], [1, 1], [c11, c22], [1, 1])
        
        # M2 = (A21+A22)B11
        # C21 += M2
        add_term([a21, a22], [1, 1], [b11], [1], [c21], [1])
        
        # M3 = A11(B21-B22)
        # C12 += M3
        add_term([a11], [1], [b21, b22], [1, -1], [c12], [1])
        
        # M4 = A22(B12-B11)
        # C11 += M4, C21 += M4
        add_term([a22], [1], [b12, b11], [1, -1], [c11, c21], [1, 1])
        
        # M5 = (A11+A12)B22
        # C11 -= M5, C12 += M5
        add_term([a11, a12], [1, 1], [b22], [1], [c11, c12], [-1, 1])
        
        # M6 = (A21-A11)(B11+B12)
        # C22 += M6
        add_term([a21, a11], [1, -1], [b11, b12], [1, 1], [c22], [1])
        
        # M7 = (A12-A22)(B21+B22)
        # C11 += M7
        add_term([a12, a22], [1, -1], [b21, b22], [1, 1], [c11], [1])

    # Helper for 1x2x2 block
    # A is at row r, cols c0, c1
    # B is at rows c0, c1 and cols d0, d1
    # C is at row r, cols d0, d1
    def add_simple_1x2(r, c0, c1, d0, d1):
        a1 = r*m + c0
        a2 = r*m + c1
        
        b11 = c0*p + d0
        b12 = c0*p + d1
        b21 = c1*p + d0
        b22 = c1*p + d1
        
        c1 = d0*n + r
        c2 = d1*n + r
        
        # C1 = A1 B11 + A2 B21
        add_term([a1], [1], [b11], [1], [c1], [1])
        add_term([a2], [1], [b21], [1], [c1], [1])
        
        # C2 = A1 B12 + A2 B22
        add_term([a1], [1], [b12], [1], [c2], [1])
        add_term([a2], [1], [b22], [1], [c2], [1])

    # Loop over 3 column blocks of B (size 2 each)
    for k in range(3):
        d0 = 2*k
        d1 = 2*k + 1
        
        # Block 1: A rows 0,1 (2x4) times B block (4x2)
        # Split A cols 0,1 and 2,3
        # Split B rows 0,1 and 2,3
        
        # Sub-block 1: A rows 0,1 cols 0,1 (2x2) times B rows 0,1 cols d0,d1 (2x2)
        add_strassen(0, 1, 0, 1, d0, d1)
        
        # Sub-block 2: A rows 0,1 cols 2,3 (2x2) times B rows 2,3 cols d0,d1 (2x2)
        add_strassen(0, 1, 2, 3, d0, d1)
        
        # Block 2: A row 2 (1x4) times B block (4x2)
        # Split A cols 0,1 and 2,3
        # Split B rows 0,1 and 2,3
        
        # Sub-block 1: A row 2 cols 0,1 (1x2) times B rows 0,1 cols d0,d1 (2x2)
        add_simple_1x2(2, 0, 1, d0, d1)
        
        # Sub-block 2: A row 2 cols 2,3 (1x2) times B rows 2,3 cols d0,d1 (2x2)
        add_simple_1x2(2, 2, 3, d0, d1)

    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return {'U': U, 'V': V, 'W': W}
