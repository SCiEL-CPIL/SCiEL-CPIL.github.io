import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # a_idx: 4 indices for A (i1j1, i1j2, i2j1, i2j2)
    # b_idx: 4 indices for B (i1k1, i1k2, i2k1, i2k2)
    # c_idx: 4 indices for C (k1i1, k1i2, k2i1, k2i2) -- Note: W is k*n+i
    # coeffs: list of 7 tuples (u_coeff, v_coeff, w_coeffs)
    # Actually, it's easier to just iterate through the 7 Strassen terms.
    
    def add_strassen(a_indices, b_indices, c_indices, start_col):
        # a_indices: [a11, a12, a21, a22]
        # b_indices: [b11, b12, b21, b22]
        # c_indices: [c11, c12, c21, c22] corresponding to (k,i)
        
        # M1 = (a11 + a22)(b11 + b22)
        U[a_indices[0]][start_col] = 1
        U[a_indices[3]][start_col] = 1
        V[b_indices[0]][start_col] = 1
        V[b_indices[3]][start_col] = 1
        W[c_indices[0]][start_col] = 1 # c11
        W[c_indices[3]][start_col] = 1 # c22
        # Wait, C11 = M1 + ...
        # C22 = M1 - ...
        # So M1 contributes to C11 and C22.
        
        # M2 = (a21 + a22) b11
        U[a_indices[2]][start_col+1] = 1
        U[a_indices[3]][start_col+1] = 1
        V[b_indices[0]][start_col+1] = 1
        W[c_indices[2]][start_col+1] = 1 # c21
        W[c_indices[3]][start_col+1] = -1 # c22
        
        # M3 = a11 (b12 - b22)
        U[a_indices[0]][start_col+2] = 1
        V[b_indices[1]][start_col+2] = 1
        V[b_indices[3]][start_col+2] = -1
        W[c_indices[1]][start_col+2] = 1 # c12
        W[c_indices[3]][start_col+2] = 1 # c22
        
        # M4 = a22 (b21 - b11)
        U[a_indices[3]][start_col+3] = 1
        V[b_indices[2]][start_col+3] = 1
        V[b_indices[0]][start_col+3] = -1
        W[c_indices[0]][start_col+3] = 1 # c11
        W[c_indices[2]][start_col+3] = 1 # c21
        
        # M5 = (a11 + a12) b22
        U[a_indices[0]][start_col+4] = 1
        U[a_indices[1]][start_col+4] = 1
        V[b_indices[3]][start_col+4] = 1
        W[c_indices[0]][start_col+4] = -1 # c11
        W[c_indices[1]][start_col+4] = 1 # c12
        
        # M6 = (a21 - a11) (b11 + b12)
        U[a_indices[2]][start_col+5] = 1
        U[a_indices[0]][start_col+5] = -1
        V[b_indices[0]][start_col+5] = 1
        V[b_indices[1]][start_col+5] = 1
        W[c_indices[3]][start_col+5] = 1 # c22
        
        # M7 = (a12 - a22) (b21 + b22)
        U[a_indices[1]][start_col+6] = 1
        U[a_indices[3]][start_col+6] = -1
        V[b_indices[2]][start_col+6] = 1
        V[b_indices[3]][start_col+6] = 1
        W[c_indices[0]][start_col+6] = 1 # c11
        W[c_indices[1]][start_col+6] = -1 # c12
        
        return start_col + 7

    def add_scalar(a_idx, b_idx, c_idx, start_col, coeff=1):
        U[a_idx][start_col] = coeff
        V[b_idx][start_col] = coeff
        W[c_idx][start_col] = coeff
        return start_col + 1

    # Block 1: A_L (cols 0,1) x B_L (rows 0,1)
    # A_L1 (2x2): i=0,1; j=0,1
    # Indices in U: i*4+j
    a11 = 0*4+0; a12 = 0*4+1; a21 = 1*4+0; a22 = 1*4+1
    a1_indices = [a11, a12, a21, a22]
    
    # B_L1 (2x2): j=0,1; k=0,1
    # Indices in V: j*4+k
    b11 = 0*4+0; b12 = 0*4+1; b21 = 1*4+0; b22 = 1*4+1
    b1_indices = [b11, b12, b21, b22]
    
    # C_L11 (2x2): k=0,1; i=0,1
    # Indices in W: k*3+i
    c11 = 0*3+0; c12 = 0*3+1; c21 = 1*3+0; c22 = 1*3+1
    c1_indices = [c11, c12, c21, c22]
    
    col = add_strassen(a1_indices, b1_indices, c1_indices, col)
    
    # B_L2 (2x2): j=0,1; k=2,3
    b11_2 = 0*4+2; b12_2 = 0*4+3; b21_2 = 1*4+2; b22_2 = 1*4+3
    b2_indices = [b11_2, b12_2, b21_2, b22_2]
    
    # C_L12 (2x2): k=2,3; i=0,1
    c11_2 = 2*3+0; c12_2 = 2*3+1; c21_2 = 3*3+0; c22_2 = 3*3+1
    c2_indices = [c11_2, c12_2, c21_2, c22_2]
    
    col = add_strassen(a1_indices, b2_indices, c2_indices, col)
    
    # A_L2 (1x2): i=2; j=0,1
    a31 = 2*4+0; a32 = 2*4+1
    
    # B_L (2x4): j=0,1; k=0,1,2,3
    # We need 8 scalar products: a31*b1k + a32*b2k for k=0..3
    
    # k=0
    col = add_scalar(a31, 0*4+0, 0*3+2, col)
    col = add_scalar(a32, 1*4+0, 0*3+2, col)
    # k=1
    col = add_scalar(a31, 0*4+1, 1*3+2, col)
    col = add_scalar(a32, 1*4+1, 1*3+2, col)
    # k=2
    col = add_scalar(a31, 0*4+2, 2*3+2, col)
    col = add_scalar(a32, 1*4+2, 2*3+2, col)
    # k=3
    col = add_scalar(a31, 0*4+3, 3*3+2, col)
    col = add_scalar(a32, 1*4+3, 3*3+2, col)
    
    # Block 2: A_R (cols 2,3) x B_R (rows 2,3)
    # A_R1 (2x2): i=0,1; j=2,3
    a11_r = 0*4+2; a12_r = 0*4+3; a21_r = 1*4+2; a22_r = 1*4+3
    a1_indices_r = [a11_r, a12_r, a21_r, a22_r]
    
    # B_R1 (2x2): j=2,3; k=0,1
    b11_r = 2*4+0; b12_r = 2*4+1; b21_r = 3*4+0; b22_r = 3*4+1
    b1_indices_r = [b11_r, b12_r, b21_r, b22_r]
    
    # C_R11 (2x2): k=0,1; i=0,1
    # Same C indices as before
    col = add_strassen(a1_indices_r, b1_indices_r, c1_indices, col)
    
    # B_R2 (2x2): j=2,3; k=2,3
    b11_r2 = 2*4+2; b12_r2 = 2*4+3; b21_r2 = 3*4+2; b22_r2 = 3*4+3
    b2_indices_r = [b11_r2, b12_r2, b21_r2, b22_r2]
    
    # C_R12 (2x2): k=2,3; i=0,1
    col = add_strassen(a1_indices_r, b2_indices_r, c2_indices, col)
    
    # A_R2 (1x2): i=2; j=2,3
    a31_r = 2*4+2; a32_r = 2*4+3
    
    # B_R (2x4): j=2,3; k=0,1,2,3
    # k=0
    col = add_scalar(a31_r, 2*4+0, 0*3+2, col)
    col = add_scalar(a32_r, 3*4+0, 0*3+2, col)
    # k=1
    col = add_scalar(a31_r, 2*4+1, 1*3+2, col)
    col = add_scalar(a32_r, 3*4+1, 1*3+2, col)
    # k=2
    col = add_scalar(a31_r, 2*4+2, 2*3+2, col)
    col = add_scalar(a32_r, 3*4+2, 2*3+2, col)
    # k=3
    col = add_scalar(a31_r, 2*4+3, 3*3+2, col)
    col = add_scalar(a32_r, 3*4+3, 3*3+2, col)
    
    return {"U": U, "V": V, "W": W}
