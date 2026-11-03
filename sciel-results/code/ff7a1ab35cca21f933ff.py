import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # a,b,c,d are A entries (row-major: 0,1,2,3)
    # e,f,g,h are B entries (row-major: 0,1,2,3)
    # C entries:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_A = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1], # f-h
        [0, 0, 1, -1], # g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    strassen_C = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, 0],  # M3
        [1, 0, 0, 0],  # M4
        [0, 1, 0, 0],  # M5
        [0, 0, 0, 1],  # M6
        [0, 0, 1, 0]   # M7
    ]
    # Wait, C coefficients need to be combined.
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # C10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C11 = 1*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    
    c_coeffs = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    col = 0
    for j in range(4):
        # A1 is A[:, 2j:2j+2]
        # A2 is A[:, 2j+2:2j+4]
        
        # B1j is B[:2, 2j:2j+2]
        # B2j is B[2:, 2j:2j+2]
        
        # Process P_j = A1 * B1j
        for k in range(7):
            # A1 indices in global U
            # A1[0,0] -> U[2j]
            # A1[0,1] -> U[2j+1]
            # A1[1,0] -> U[4+2j]
            # A1[1,1] -> U[4+2j+1]
            
            a_idx = [2j, 2j+1, 4+2j, 4+2j+1]
            for i, idx in enumerate(a_idx):
                U[idx, col] = strassen_A[k][i]
                
            # B1j indices in global V
            # B1j[0,0] -> V[0*8 + 2j]
            # B1j[0,1] -> V[0*8 + 2j+1]
            # B1j[1,0] -> V[1*8 + 2j]
            # B1j[1,1] -> V[1*8 + 2j+1]
            
            b_idx = [2j, 2j+1, 8+2j, 8+2j+1]
            for i, idx in enumerate(b_idx):
                V[idx, col] = strassen_B[k][i]
                
            # C_j indices in global W
            # C_j[0,0] -> W[2j*2 + 0] = W[4j]
            # C_j[0,1] -> W[(2j+1)*2 + 0] = W[4j+2]
            # C_j[1,0] -> W[2j*2 + 1] = W[4j+1]
            # C_j[1,1] -> W[(2j+1)*2 + 1] = W[4j+3]
            
            c_idx = [4j, 4j+2, 4j+1, 4j+3]
            for i, idx in enumerate(c_idx):
                W[idx, col] = c_coeffs[i][k]
                
            col += 1
            
        # Process Q_j = A2 * B2j
        for k in range(7):
            # A2 indices in global U
            # A2[0,0] -> U[2j+2]
            # A2[0,1] -> U[2j+3]
            # A2[1,0] -> U[4+2j+2]
            # A2[1,1] -> U[4+2j+3]
            
            a_idx = [2j+2, 2j+3, 4+2j+2, 4+2j+3]
            for i, idx in enumerate(a_idx):
                U[idx, col] = strassen_A[k][i]
                
            # B2j indices in global V
            # B2j[0,0] -> V[2*8 + 2j]
            # B2j[0,1] -> V[2*8 + 2j+1]
            # B2j[1,0] -> V[3*8 + 2j]
            # B2j[1,1] -> V[3*8 + 2j+1]
            
            b_idx = [16+2j, 16+2j+1, 24+2j, 24+2j+1]
            for i, idx in enumerate(b_idx):
                V[idx, col] = strassen_B[k][i]
                
            # C_j indices in global W (same as before)
            c_idx = [4j, 4j+2, 4j+1, 4j+3]
            for i, idx in enumerate(c_idx):
                W[idx, col] = c_coeffs[i][k]
                
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
