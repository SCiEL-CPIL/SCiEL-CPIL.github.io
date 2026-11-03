import numpy as np

def construct():
    n, m, p = 2, 3, 8
    u_dim, v_dim, w_dim = n*m, m*p, p*n
    
    U = np.zeros((u_dim, 44), dtype=float)
    V = np.zeros((v_dim, 44), dtype=float)
    W = np.zeros((w_dim, 44), dtype=float)
    
    # 7 Strassen terms for each of 4 blocks (2x2 * 2x2)
    for b in range(4):
        j0 = 2*b
        j1 = 2*b+1
        k0 = 2*b
        k1 = 2*b+1
        
        # Row indices in U (2x3 matrix, row-major: i*m+j)
        r_a = 0*m + j0   # A[0, j0] = a
        r_b = 0*m + j1   # A[0, j1] = b
        r_c = 1*m + j0   # A[1, j0] = c
        r_d = 1*m + j1   # A[1, j1] = d
        
        # Row indices in V (3x8 matrix, row-major: j*p+k)
        r_e = j0*p + k0   # B[j0, k0] = e
        r_f = j0*p + k1   # B[j0, k1] = f
        r_g = j1*p + k0   # B[j1, k0] = g
        r_h = j1*p + k1   # B[j1, k1] = h
        
        # Column indices in W (8x2 matrix, row-major: k*n+i)
        c_r00 = k0*n + 0   # C[0, k0]
        c_r01 = k1*n + 0   # C[0, k1]
        c_r10 = k0*n + 1   # C[1, k0]
        c_r11 = k1*n + 1   # C[1, k1]
        
        base = 7*b
        
        # P1 = (a+d)(e+h)
        U[r_a, base+0] = 1; U[r_d, base+0] = 1
        V[r_e, base+0] = 1; V[r_h, base+0] = 1
        W[c_r00, base+0] = 1; W[c_r11, base+0] = 1
        
        # P2 = (c+d)e
        U[r_c, base+1] = 1; U[r_d, base+1] = 1
        V[r_e, base+1] = 1
        W[c_r10, base+1] = 1; W[c_r11, base+1] = -1
        
        # P3 = a(f-h)
        U[r_a, base+2] = 1
        V[r_f, base+2] = 1; V[r_h, base+2] = -1
        W[c_r01, base+2] = 1; W[c_r11, base+2] = 1
        
        # P4 = d(g-e)
        U[r_d, base+3] = 1
        V[r_g, base+3] = 1; V[r_e, base+3] = -1
        W[c_r00, base+3] = 1; W[c_r10, base+3] = 1
        
        # P5 = (a+b)h
        U[r_a, base+4] = 1; U[r_b, base+4] = 1
        V[r_h, base+4] = 1
        W[c_r00, base+4] = -1; W[c_r01, base+4] = 1
        
        # P6 = (c-a)(e+f)
        U[r_c, base+5] = 1; U[r_a, base+5] = -1
        V[r_e, base+5] = 1; V[r_f, base+5] = 1
        W[c_r11, base+5] = 1
        
        # P7 = (b-d)(g+h)
        U[r_b, base+6] = 1; U[r_d, base+6] = -1
        V[r_g, base+6] = 1; V[r_h, base+6] = 1
        W[c_r00, base+6] = 1
    
    # 16 schoolbook terms for the remaining cross terms
    # A is 2x3, B is 3x8. The Strassen blocks cover A[0:2, 0:2]*B[0:2, 0:2], etc.
    # Actually, let me reconsider the structure.
    
    # For 2x3 * 3x8:
    # C[0, k] = sum_j A[0,j]*B[j,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1, k] = sum_j A[1,j]*B[j,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # The Strassen decomposition I wrote above assumes A is 2x2 and B is 2x2.
    # For 2x3 * 3x8, we need a different approach.
    
    # Let me redo this properly.
    # We can split the 3 columns of A into 1+2, or use a different blocking.
    
    # Actually, let's use the standard approach:
    # Split A (2x3) into [A_left (2x1), A_right (2x2)]
    # Split B (3x8) into [B_top (1x8), B_bottom (2x8)]
    # Then C = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 2x1 * 1x8 = 2x8, rank 16 (schoolbook)
    # A_right * B_bottom is 2x2 * 2x8, can use Strassen on 2x2 blocks
    
    # For A_right (2x2) * B_bottom (2x8):
    # Split B_bottom into 4 blocks of 2x2
    # Use Strassen on each 2x2 block: 7 terms each, 4 blocks = 28 terms
    # Total: 16 + 28 = 44 terms
    
    # Let me rebuild this properly.
    
    U = np.zeros((u_dim, 44), dtype=float)
    V = np.zeros((v_dim, 44), dtype=float)
    W = np.zeros((w_dim, 44), dtype=float)
    
    # Part 1: A_left (2x1) * B_top (1x8)
    # A_left = [A[0,0], A[1,0]], B_top = B[0, :]
    # C[0, k] += A[0,0] * B[0, k]
    # C[1, k] += A[1,0] * B[0, k]
    for k in range(8):
        U[0*m+0, k] = 1        # A[0,0]
        V[0*p+k, k] = 1        # B[0,k]
        W[k*n+0, k] = 1        # C[0,k]
    for k in range(8):
        U[1*m+0, 8+k] = 1      # A[1,0]
        V[0*p+k, 8+k] = 1      # B[0,k]
        W[k*n+1, 8+k] = 1      # C[1,k]
    
    # Part 2: A_right (2x2) * B_bottom (2x8)
    # A_right = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B_bottom = B[1:3, :] (2x8)
    # Split B_bottom into 4 blocks of 2x2
    for b in range(4):
        k0 = 2*b
        k1 = 2*b+1
        base = 16 + 7*b
        
        # A_right entries
        r_a = 0*m + 1   # A[0,1]
        r_b = 0*m + 2   # A[0,2]
        r_c = 1*m + 1   # A[1,1]
        r_d = 1*m + 2   # A[1,2]
        
        # B_bottom entries (rows 1 and 2 of B)
        r_e = 1*p + k0   # B[1, k0]
        r_f = 1*p + k1   # B[1, k1]
        r_g = 2*p + k0   # B[2, k0]
        r_h = 2*p + k1   # B[2, k1]
        
        # Output columns
        c_r00 = k0*n + 0
        c_r01 = k1*n + 0
        c_r10 = k0*n + 1
        c_r11 = k1*n + 1
        
        # P1 = (a+d)(e+h)
        U[r_a, base+0] = 1; U[r_d, base+0] = 1
        V[r_e, base+0] = 1; V[r_h, base+0] = 1
        W[c_r00, base+0] = 1; W[c_r11, base+0] = 1
        
        # P2 = (c+d)e
        U[r_c, base+1] = 1; U[r_d, base+1] = 1
        V[r_e, base+1] = 1
        W[c_r10, base+1] = 1; W[c_r11, base+1] = -1
        
        # P3 = a(f-h)
        U[r_a, base+2] = 1
        V[r_f, base+2] = 1; V[r_h, base+2] = -1
        W[c_r01, base+2] = 1; W[c_r11, base+2] = 1
        
        # P4 = d(g-e)
        U[r_d, base+3] = 1
        V[r_g, base+3] = 1; V[r_e, base+3] = -1
        W[c_r00, base+3] = 1; W[c_r10, base+3] = 1
        
        # P5 = (a+b)h
        U[r_a, base+4] = 1; U[r_b, base+4] = 1
        V[r_h, base+4] = 1
        W[c_r00, base+4] = -1; W[c_r01, base+4] = 1
        
        # P6 = (c-a)(e+f)
        U[r_c, base+5] = 1; U[r_a, base+5] = -1
        V[r_e, base+5] = 1; V[r_f, base+5] = 1
        W[c_r11, base+5] = 1
        
        # P7 = (b-d)(g+h)
        U[r_b, base+6] = 1; U[r_d, base+6] = -1
        V[r_g, base+6] = 1; V[r_h, base+6] = 1
        W[c_r00, base+6] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
