"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # Output C is 2x9, indexed as W[k*n+i, r] = W[2*k+i, r]
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (18, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_rows.append(u_vec[:])
        V_rows.append(v_vec[:])
        W_rows.append(w_vec[:])
    
    # For each block of 2 columns in B (columns k, k+1 for k=0,2,4,6)
    # We use Strassen's algorithm for 2x2 x 2x2
    # B indices: B[j, k] is at position j*p + k = j*9 + k
    
    for block_start in [0, 2, 4, 6]:
        k1 = block_start
        k2 = block_start + 1
        
        # B column indices for the 2x2 block
        # B[0,k1] at 0*9+k1, B[1,k1] at 1*9+k1
        # B[0,k2] at 0*9+k2, B[1,k2] at 1*9+k2
        b00 = k1      # B[0,k1]
        b10 = 9 + k1  # B[1,k1]
        b01 = k2      # B[0,k2]
        b11 = 9 + k2  # B[1,k2]
        
        # Output C indices for this block
        # C[0,k1] at W[2*k1+0], C[1,k1] at W[2*k1+1]
        # C[0,k2] at W[2*k2+0], C[1,k2] at W[2*k2+1]
        c00 = 2*k1 + 0
        c10 = 2*k1 + 1
        c01 = 2*k2 + 0
        c11 = 2*k2 + 1
        
        # Strassen's 7 terms for A*B where A is 2x2, B is 2x2
        # A indices: A[i,j] at U row i*m+j = i*2+j
        # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
        
        # M1 = (a+d)(e+h) where e=B[0,k1], h=B[1,k2]
        # e = B[0,k1] at b00, h = B[1,k2] at b11
        u_M1 = [1, 0, 0, 1]  # a+d
        v_M1 = [0]*18
        v_M1[b00] = 1
        v_M1[b11] = 1
        w_M1 = [0]*18
        w_M1[c00] = 1
        w_M1[c11] = 1
        w_M1[c01] = -1
        w_M1[c10] = 1
        add_term(u_M1, v_M1, w_M1)
        
        # M2 = (c+d)e where e=B[0,k1]
        u_M2 = [0, 0, 1, 1]  # c+d
        v_M2 = [0]*18
        v_M2[b00] = 1
        w_M2 = [0]*18
        w_M2[c10] = 1
        w_M2[c11] = -1
        add_term(u_M2, v_M2, w_M2)
        
        # M3 = a(f-h) where f=B[1,k1], h=B[1,k2]
        u_M3 = [1, 0, 0, 0]  # a
        v_M3 = [0]*18
        v_M3[b10] = 1
        v_M3[b11] = -1
        w_M3 = [0]*18
        w_M3[c01] = 1
        w_M3[c11] = 1
        add_term(u_M3, v_M3, w_M3)
        
        # M4 = d(g-e) where g=B[1,k1], e=B[0,k1]
        u_M4 = [0, 0, 0, 1]  # d
        v_M4 = [0]*18
        v_M4[b10] = 1
        v_M4[b00] = -1
        w_M4 = [0]*18
        w_M4[c00] = 1
        w_M4[c10] = 1
        add_term(u_M4, v_M4, w_M4)
        
        # M5 = (a+b)h where h=B[1,k2]
        u_M5 = [1, 1, 0, 0]  # a+b
        v_M5 = [0]*18
        v_M5[b11] = 1
        w_M5 = [0]*18
        w_M5[c00] = -1
        w_M5[c01] = 1
        add_term(u_M5, v_M5, w_M5)
        
        # M6 = (c-a)(e+f) where e=B[0,k1], f=B[1,k1]
        u_M6 = [-1, 0, 1, 0]  # c-a
        v_M6 = [0]*18
        v_M6[b00] = 1
        v_M6[b10] = 1
        w_M6 = [0]*18
        w_M6[c11] = 1
        add_term(u_M6, v_M6, w_M6)
        
        # M7 = (b-d)(g+h) where g=B[1,k1], h=B[1,k2]
        u_M7 = [0, 1, 0, -1]  # b-d
        v_M7 = [0]*18
        v_M7[b10] = 1
        v_M7[b11] = 1
        w_M7 = [0]*18
        w_M7[c00] = 1
        w_M7[c01] = 1
        add_term(u_M7, v_M7, w_M7)
    
    # Last column: k=8, schoolbook
    k = 8
    # B[0,8] at 0*9+8=8, B[1,8] at 1*9+8=17
    b0 = 8
    b1 = 17
    # C[0,8] at W[2*8+0]=16, C[1,8] at W[2*8+1]=17
    c0 = 16
    c1 = 17
    
    # Term 1: A[0,0]*B[0,8] -> C[0,8]
    u1 = [1, 0, 0, 0]
    v1 = [0]*18
    v1[b0] = 1
    w1 = [0]*18
    w1[c0] = 1
    add_term(u1, v1, w1)
    
    # Term 2: A[0,1]*B[1,8] -> C[0,8]
    u2 = [0, 1, 0, 0]
    v2 = [0]*18
    v2[b1] = 1
    w2 = [0]*18
    w2[c0] = 1
    add_term(u2, v2, w2)
    
    # Term 3: A[1,0]*B[0,8] -> C[1,8]
    u3 = [0, 0, 1, 0]
    v3 = [0]*18
    v3[b0] = 1
    w3 = [0]*18
    w3[c1] = 1
    add_term(u3, v3, w3)
    
    # Term 4: A[1,1]*B[1,8] -> C[1,8]
    u4 = [0, 0, 0, 1]
    v4 = [0]*18
    v4[b1] = 1
    w4 = [0]*18
    w4[c1] = 1
    add_term(u4, v4, w4)
    
    rank = len(U_rows)
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
