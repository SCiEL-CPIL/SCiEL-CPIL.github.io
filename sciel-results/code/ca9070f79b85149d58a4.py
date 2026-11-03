import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    # Strassen 2x2x2 decomposition
    # U: (4,7) indexed by i*2+j, V: (4,7) indexed by j*2+k, W: (4,7) indexed by k*2+i
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1
    
    # Verify Strassen 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Fall back to schoolbook for 4x4x8
                    rank = n * m * p
                    U = np.zeros((n*m, rank), dtype=float)
                    V = np.zeros((m*p, rank), dtype=float)
                    W = np.zeros((p*n, rank), dtype=float)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj, col] = 1
                                V[jj*p+kk, col] = 1
                                W[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Define Strassen block coefficients for 4x4x4
    # A_blocks: (0,0), (0,1), (1,0), (1,1)
    # B_blocks: (0,0), (0,1), (1,0), (1,1)
    # C_blocks: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    strassen_defs = [
        # Mk: (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
         {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Build 4x4x4 decomposition (rank 49)
    rank4 = 49
    U4 = np.zeros((16, rank4), dtype=float)
    V4 = np.zeros((16, rank4), dtype=float)
    W4 = np.zeros((16, rank4), dtype=float)
    
    for mk in range(7):
        A_coeffs, B_coeffs, C_coeffs = strassen_defs[mk]
        for r in range(7):
            col = mk * 7 + r
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+j2, r]
                        v_val = V2[j2*2+k2, r]
                        w_val = W2[k2*2+i2, r]
                        
                        if abs(u_val) < 1e-15 and abs(v_val) < 1e-15 and abs(w_val) < 1e-15:
                            continue
                        
                        # U4: index I*4+J where I = 2*br + i2, J = 2*bc + j2
                        for (br, bc), ca in A_coeffs.items():
                            I = 2*br + i2
                            J = 2*bc + j2
                            U4[I*4+J, col] += ca * u_val
                        
                        # V4: index J*4+K where J = 2*br + j2, K = 2*bc + k2
                        for (br, bc), cb in B_coeffs.items():
                            J = 2*br + j2
                            K = 2*bc + k2
                            V4[J*4+K, col] += cb * v_val
                        
                        # W4: index K*4+I where K = 2*br + k2, I = 2*bc + i2
                        for (br, bc), cc in C_coeffs.items():
                            K = 2*br + k2
                            I = 2*bc + i2
                            W4[K*4+I, col] += cc * w_val
    
    # Verify 4x4x4
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                if abs(val - 1.0) > 1e-8:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj, col] = 1
                    V[jj*p+kk, col] = 1
                    W[kk*n+ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Extend to 4x4x8
    # B is 4x8, split into B_left (4x4) and B_right (4x4)
    # C is 4x8
    # U: (16, rank), V: (32, rank), W: (32, rank)
    # U index: i*4+j, V index: j*8+k, W index: k*4+i
    
    rank8 = 98
    U8 = np.zeros((16, rank8), dtype=float)
    V8 = np.zeros((32, rank8), dtype=float)
    W8 = np.zeros((32, rank8), dtype=float)
    
    # Left half: columns 0..48
    U8[:, :49] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+k, :49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W8[k*4+i, :49] = W4[k*4+i, :]
    
    # Right half: columns 49..97
    U8[:, 49:] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+(k+4), 49:] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W8[(k+4)*4+i, 49:] = W4[k*4+i, :]
    
    # Verify 4x4x8
    valid8 = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = np.sum(U8[i*4+j, :] * V8[j*8+k, :] * W8[k*4+i, :])
                if abs(val - 1.0) > 1e-8:
                    valid8 = False
                    break
            if not valid8:
                break
        if not valid8:
            break
    
    if not valid8:
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj, col] = 1
                    V[jj*p+kk, col] = 1
                    W[kk*n+ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
