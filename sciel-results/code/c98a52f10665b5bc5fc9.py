import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36: 35 from Strassen on 2x2x2x10, 1 from 2x1x1x10
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # A1 uses j in {0, 1}
    # B1 uses j in {0, 1}
    # Split B1 into 5 blocks of 2x2
    # Block b uses cols 2b, 2b+1
    
    col_idx = 0
    for b in range(5):
        # Strassen steps s in 0..6
        for s in range(7):
            r = col_idx
            col_idx += 1
            
            # Define A coefficients (2x2)
            # M0: (A00+A11)(B00+B11)
            # M1: (A10+A11)B00
            # M2: A00(B10+B11)
            # M3: A11(B01-B11)
            # M4: (A00+A01)B11
            # M5: (A10-A00)(B00+B01)
            # M6: (A01-A11)(B10+B00)
            
            A_coeffs = [
                [1, 0, 0, 1], # M0: A00, A01, A10, A11
                [0, 0, 1, 1], # M1
                [1, 0, 0, 0], # M2
                [0, 0, 0, 1], # M3
                [1, 1, 0, 0], # M4
                [-1, 0, 1, 0], # M5
                [0, 1, 0, -1]  # M6
            ]
            
            B_coeffs = [
                [1, 1, 1, 1], # M0: B00, B01, B10, B11
                [1, 0, 0, 0], # M1
                [0, 0, 1, 1], # M2
                [0, 1, 0, -1], # M3
                [0, 0, 0, 1], # M4
                [1, 1, 0, 0], # M5
                [1, 0, 1, 0]  # M6
            ]
            
            # Fill U
            # U[i*m + j, r]
            # A_coeffs[s] is [c00, c01, c10, c11] for A_{00}, A_{01}, A_{10}, A_{11}
            # i in 0,1; j in 0,1
            U[0*m + 0][r] = A_coeffs[s][0]
            U[0*m + 1][r] = A_coeffs[s][1]
            U[1*m + 0][r] = A_coeffs[s][2]
            U[1*m + 1][r] = A_coeffs[s][3]
            
            # Fill V
            # V[j*p + k, r]
            # B_coeffs[s] is [c00, c01, c10, c11] for B_{00}, B_{01}, B_{10}, B_{11}
            # j in 0,1; k in 2b, 2b+1
            # B_{00} is B[0, 2b], B_{01} is B[0, 2b+1]
            # B_{10} is B[1, 2b], B_{11} is B[1, 2b+1]
            
            V[0*p + (2*b)    ][r] = B_coeffs[s][0]
            V[0*p + (2*b + 1)][r] = B_coeffs[s][1]
            V[1*p + (2*b)    ][r] = B_coeffs[s][2]
            V[1*p + (2*b + 1)][r] = B_coeffs[s][3]
            
            # Fill W
            # W[k*n + i, r]
            # C_ij = sum_s K_ij_s M_s
            # K matrix for Strassen:
            # C00 = M0 + M3 - M4 + M6
            # C01 = M1 + M4
            # C10 = M2 + M5
            # C11 = M0 - M1 + M2 + M5
            
            K = [
                [1, 1, 0, 1, -1, 0, 1], # C00
                [0, 1, 0, 0, 1, 0, 0],  # C01
                [0, 0, 1, 0, 0, 1, 0],  # C10
                [1, -1, 1, 0, 0, 1, 0]  # C11
            ]
            
            # i in 0,1; k in 2b, 2b+1
            # K[i_local, k_local, s]
            
            W[(2*b)    *n + 0][r] = K[0][s] # i=0, k=0
            W[(2*b + 1)*n + 0][r] = K[1][s] # i=0, k=1
            W[(2*b)    *n + 1][r] = K[2][s] # i=1, k=0
            W[(2*b + 1)*n + 1][r] = K[3][s] # i=1, k=1

    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 uses j=2
    # B2 uses j=2
    # Rank 1
    r = col_idx
    col_idx += 1
    
    # U: A2_{i,2} coeff 1
    U[0*m + 2][r] = 1
    U[1*m + 2][r] = 1
    
    # V: B2_{2,k} coeff 1
    for k in range(p):
        V[2*p + k][r] = 1
        
    # W: C_{i,k} coeff 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][r] = 1

    return {"U": U, "V": V, "W": W}
