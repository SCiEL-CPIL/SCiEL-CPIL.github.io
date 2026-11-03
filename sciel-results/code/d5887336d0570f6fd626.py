import numpy as np

def construct():
    # Base case: 2x2x2 Strassen decomposition
    # U[0:2, 0:2], V[0:2, 0:2], W[0:2, 0:2]
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Strassen 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Note: Indices in standard math are 1-based, here 0-based.
    # A11 -> A[0,0], A22 -> A[1,1], etc.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1
    U2[0][0] = 1; U2[3][0] = 1 # A00, A11
    V2[0][0] = 1; V2[3][0] = 1 # B00, B11
    W2[0][0] = 1; W2[3][0] = 1 # C00, C11
    
    # c=1: M2
    U2[2][1] = 1; U2[3][1] = 1 # A10, A11
    V2[0][1] = 1 # B00
    W2[2][1] = 1; W2[3][1] = -1 # C10, C11
    
    # c=2: M3
    U2[0][2] = 1 # A00
    V2[1][2] = 1; V2[3][2] = -1 # B01, B11
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # c=3: M4
    U2[3][3] = 1 # A11
    V2[2][3] = 1; V2[0][3] = -1 # B10, B00
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # c=4: M5
    U2[0][4] = 1; U2[1][4] = 1 # A00, A01
    V2[3][4] = 1 # B11
    W2[0][4] = -1; W2[1][4] = 1 # C00, C01
    
    # c=5: M6
    U2[2][5] = 1; U2[0][5] = -1 # A10, A00
    V2[0][5] = 1; V2[1][5] = 1 # B00, B01
    W2[3][5] = 1 # C11
    
    # c=6: M7
    U2[1][6] = 1; U2[3][6] = -1 # A01, A11
    V2[2][6] = 1; V2[3][6] = 1 # B10, B11
    W2[0][6] = 1 # C00

    # Construct 4x4x4 factors
    # Rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Coefficients for top level Strassen (2x2 blocks of 2x2 matrices)
    # A_c coefficients for blocks (i', j')
    coeff_A = [
        {(0,0): 1, (1,1): 1}, # c=0
        {(1,0): 1, (1,1): 1}, # c=1
        {(0,0): 1},           # c=2
        {(1,1): 1},           # c=3
        {(0,0): 1, (0,1): 1}, # c=4
        {(1,0): 1, (0,0): -1},# c=5
        {(0,1): 1, (1,1): -1} # c=6
    ]
    
    # B_c coefficients for blocks (j', k')
    coeff_B = [
        {(0,0): 1, (1,1): 1}, # c=0
        {(0,0): 1},           # c=1
        {(0,1): 1, (1,1): -1},# c=2
        {(1,0): 1, (0,0): -1},# c=3
        {(1,1): 1},           # c=4
        {(0,0): 1, (0,1): 1}, # c=5
        {(1,0): 1, (1,1): 1}  # c=6
    ]
    
    # C_c coefficients for blocks (k', i')
    coeff_C = [
        {(0,0): 1, (1,1): 1}, # c=0
        {(1,0): 1, (1,1): -1},# c=1
        {(0,1): 1, (1,1): 1}, # c=2
        {(0,0): 1, (1,0): 1}, # c=3
        {(0,0): -1, (0,1): 1},# c=4
        {(1,1): 1},           # c=5
        {(0,0): 1}            # c=6
    ]
    
    for c in range(7):
        for c2 in range(7):
            col = c * 7 + c2
            # Iterate over blocks
            for i_prime in range(2):
                for j_prime in range(2):
                    if (i_prime, j_prime) in coeff_A[c]:
                        val_A = coeff_A[c][(i_prime, j_prime)]
                        # Map to U4 indices
                        # i = 2*i_prime + i'', j = 2*j_prime + j''
                        for i_double in range(2):
                            for j_double in range(2):
                                u_val = U2[i_double*2 + j_double][c2]
                                if u_val != 0:
                                    i = 2*i_prime + i_double
                                    j = 2*j_prime + j_double
                                    U4[i*4 + j][col] += val_A * u_val
                                    
            for j_prime in range(2):
                for k_prime in range(2):
                    if (j_prime, k_prime) in coeff_B[c]:
                        val_B = coeff_B[c][(j_prime, k_prime)]
                        for j_double in range(2):
                            for k_double in range(2):
                                v_val = V2[j_double*2 + k_double][c2]
                                if v_val != 0:
                                    j = 2*j_prime + j_double
                                    k = 2*k_prime + k_double
                                    V4[j*4 + k][col] += val_B * v_val
                                    
            for k_prime in range(2):
                for i_prime in range(2):
                    if (k_prime, i_prime) in coeff_C[c]:
                        val_C = coeff_C[c][(k_prime, i_prime)]
                        for k_double in range(2):
                            for i_double in range(2):
                                w_val = W2[k_double*2 + i_double][c2]
                                if w_val != 0:
                                    k = 2*k_prime + k_double
                                    i = 2*i_prime + i_double
                                    W4[k*4 + i][col] += val_C * w_val

    # Construct 4x4x8 factors
    # Rank 98
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Copy 1: k in 0..3
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                V[j*4 + k][r] = V4[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][r] = W4[k*4 + i][r]
                
    # Copy 2: k in 4..7
    for r in range(49, 98):
        r4 = r - 49
        for i in range(16):
            U[i][r] = U4[i][r4]
        for j in range(4):
            for k in range(4):
                V[j*4 + (k+4)][r] = V4[j*4 + k][r4]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i][r] = W4[k*4 + i][r4]

    return {"U": U, "V": V, "W": W}
