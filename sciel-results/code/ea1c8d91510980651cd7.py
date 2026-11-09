import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 49 (for 4x4x4) + 16 (for 4x4x1) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Generate 49 terms for 4x4x4 using Strassen
    # 2x2 Strassen terms (u, v, w)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [-1, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Strassen linear combinations for 4x4
    # A_k and B_k are 2x2 matrices of blocks.
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B01-B11)
    # k=3: M4 = A11(B10-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    # Define the linear forms for A and B blocks
    # Block indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    A_forms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    B_forms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, -1, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Contribution of M_k to C blocks
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M4 + M6
    C_forms = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11(-)
        [0, 1, 0, 0], # M3 -> C01
        [-1, 0, 1, 1], # M4 -> C00(-), C10, C11
        [0, 1, 0, -1], # M5 -> C01, C11(-)
        [1, 0, 0, 1], # M6 -> C00, C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    col = 0
    for k in range(7):
        a_form = A_forms[k]
        b_form = B_forms[k]
        c_form = C_forms[k]
        
        for s in range(7):
            u_2, v_2, w_2 = terms_2x2[s]
            
            # Construct u_4 (length 16)
            u_4 = [0] * 16
            for bi in range(4):
                if a_form[bi] != 0:
                    # Block bi of A corresponds to rows bi//2, cols bi%2
                    # In u_4, index is i*m + j
                    # Block 00: i=0, j=0 -> 0..3
                    # Block 01: i=0, j=1 -> 4..7
                    # Block 10: i=1, j=0 -> 8..11
                    # Block 11: i=1, j=1 -> 12..15
                    base = bi * 4
                    for t in range(4):
                        u_4[base + t] = a_form[bi] * u_2[t]
                        
            # Construct v_4 (length 16)
            v_4 = [0] * 16
            for bi in range(4):
                if b_form[bi] != 0:
                    base = bi * 4
                    for t in range(4):
                        v_4[base + t] = b_form[bi] * v_2[t]
                        
            # Construct w_4 (length 16)
            # w is for C. C is 4x4.
            # w_4 index is k*n + i.
            # Block 00: k=0, i=0 -> 0..3
            # Block 01: k=1, i=0 -> 4..7
            # Block 10: k=0, i=1 -> 8..11
            # Block 11: k=1, i=1 -> 12..15
            # Wait, w index is k*n + i.
            # Block 00 (C00): k=0, i=0. Indices 0,1,2,3?
            # C00 is a 2x2 matrix.
            # Elements C_{00}, C_{01}, C_{10}, C_{11} of the block.
            # In the full C matrix:
            # C_{0,0}, C_{0,1}, C_{1,0}, C_{1,1}
            # w index for C_{i,k} is k*4 + i.
            # For block 00 (i in 0,1; k in 0,1):
            # C_{0,0} -> 0*4+0 = 0
            # C_{0,1} -> 1*4+0 = 4
            # C_{1,0} -> 0*4+1 = 1
            # C_{1,1} -> 1*4+1 = 5
            # This mapping is tricky.
            
            # Let's map block bi of C to w_4 indices.
            # bi=0 (C00): i in {0,1}, k in {0,1}
            #   w_idx for (i=0,k=0) is 0
            #   w_idx for (i=1,k=0) is 1
            #   w_idx for (i=0,k=1) is 4
            #   w_idx for (i=1,k=1) is 5
            #   So w_4[0]=w2[0], w_4[1]=w2[1], w_4[4]=w2[2], w_4[5]=w2[3]
            
            # bi=1 (C01): i in {0,1}, k in {2,3}
            #   w_idx for (i=0,k=2) is 8
            #   w_idx for (i=1,k=2) is 9
            #   w_idx for (i=0,k=3) is 12
            #   w_idx for (i=1,k=3) is 13
            
            # bi=2 (C10): i in {2,3}, k in {0,1}
            #   w_idx for (i=2,k=0) is 2
            #   w_idx for (i=3,k=0) is 3
            #   w_idx for (i=2,k=1) is 6
            #   w_idx for (i=3,k=1) is 7
            
            # bi=3 (C11): i in {2,3}, k in {2,3}
            #   w_idx for (i=2,k=2) is 10
            #   w_idx for (i=3,k=2) is 11
            #   w_idx for (i=2,k=3) is 14
            #   w_idx for (i=3,k=3) is 15
            
            w_4 = [0] * 16
            if c_form[0] != 0: # Block 00
                w_4[0] = c_form[0] * w_2[0]
                w_4[1] = c_form[0] * w_2[1]
                w_4[4] = c_form[0] * w_2[2]
                w_4[5] = c_form[0] * w_2[3]
            if c_form[1] != 0: # Block 01
                w_4[8] = c_form[1] * w_2[0]
                w_4[9] = c_form[1] * w_2[1]
                w_4[12] = c_form[1] * w_2[2]
                w_4[13] = c_form[1] * w_2[3]
            if c_form[2] != 0: # Block 10
                w_4[2] = c_form[2] * w_2[0]
                w_4[3] = c_form[2] * w_2[1]
                w_4[6] = c_form[2] * w_2[2]
                w_4[7] = c_form[2] * w_2[3]
            if c_form[3] != 0: # Block 11
                w_4[10] = c_form[3] * w_2[0]
                w_4[11] = c_form[3] * w_2[1]
                w_4[14] = c_form[3] * w_2[2]
                w_4[15] = c_form[3] * w_2[3]
                
            # Fill U, V, W
            for i in range(16):
                U[i][col] = u_4[i]
                V[i][col] = v_4[i]
                W[i][col] = w_4[i]
            col += 1
            
    # 2. Add 16 terms for the 5th column (k=4)
    # C_{i4} = sum_j A_{ij} B_{j4}
    for i in range(4):
        for j in range(4):
            # U[i*m+j] = 1
            U[i * 4 + j][col] = 1
            # V[j*p+4] = 1
            V[j * 5 + 4][col] = 1
            # W[4*n+i] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
