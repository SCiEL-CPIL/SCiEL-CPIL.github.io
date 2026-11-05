import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    # We will build M_4,8,8 using Kronecker products and restrict to 3,5,7.
    
    # 1. M_1,2,2 (Rank 2)
    # n=1, m=2, p=2
    n1, m1, p1 = 1, 2, 2
    R1 = 2
    U1 = [[0]*R1 for _ in range(n1*m1)]
    V1 = [[0]*R1 for _ in range(m1*p1)]
    W1 = [[0]*R1 for _ in range(p1*n1)]
    
    # U[0, j, r] = delta(j, r)
    # V[j, k, r] = delta(j, r)
    # W[k, 0, r] = 1
    for j in range(m1):
        for r in range(R1):
            if j == r:
                U1[j][r] = 1
                V1[j*p1 + 0][r] = 1 # Wait, V index is j*p + k. Here p=2.
                # V[j, k, r] = delta(j, r). So for all k, V[j, k, r] = delta(j, r).
                # So V[j*p1 + k][r] = 1 if j==r else 0.
                for k in range(p1):
                    V1[j*p1 + k][r] = 1
                W1[0*n1 + 0][r] = 1 # W index is k*n + i. i=0.
    
    # 2. M_2,2,2 (Strassen, Rank 7)
    n2, m2, p2 = 2, 2, 2
    R2 = 7
    U2 = [[0]*R2 for _ in range(n2*m2)]
    V2 = [[0]*R2 for _ in range(m2*p2)]
    W2 = [[0]*R2 for _ in range(p2*n2)]
    
    # Strassen factors
    # U: 4x7
    # V: 4x7
    # W: 4x7
    # Indices: i*2+j, j*2+k, k*2+i
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2 - M5 - M7
    # C22 = M2 - M3 + M6 + M7
    
    # We need U, V, W such that sum U V W = T
    # T[i, j, k] = 1
    
    # Let's define the 7 terms.
    # Term r corresponds to M_r.
    
    # M1: (A11+A22)(B11+B22)
    # U: A11, A22 coeffs.
    # U[0,0,0]=1 (A11), U[3,0,0]=1 (A22)
    # V: B11, B22 coeffs.
    # V[0,0,0]=1 (B11), V[3,0,0]=1 (B22)
    # W: C11, C21 coeffs.
    # W[0,0,0]=1 (C11), W[2,0,0]=1 (C21)
    
    def set_strassen(r, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (i, j, val)
        # v_coeffs: list of (j, k, val)
        # w_coeffs: list of (k, i, val)
        for i, j, val in u_coeffs:
            U2[i*m2+j][r] = val
        for j, k, val in v_coeffs:
            V2[j*p2+k][r] = val
        for k, i, val in w_coeffs:
            W2[k*n2+i][r] = val
            
    # M1
    set_strassen(0, [(0,0,1), (3,0,1)], [(0,0,1), (3,0,1)], [(0,0,1), (2,0,1)])
    # M2
    set_strassen(1, [(2,0,1), (3,0,1)], [(0,0,1)], [(2,0,1), (3,0,1)]) # C21, C22
    # M3
    set_strassen(2, [(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1), (3,0,1)]) # C12, C22
    # M4
    set_strassen(3, [(3,0,1)], [(0,1,1), (0,0,-1)], [(1,0,1), (0,0,-1)]) # C12, C11
    # M5
    set_strassen(4, [(0,0,1), (0,1,1)], [(3,0,1)], [(0,0,1), (2,0,-1)]) # C11, C21
    # M6
    set_strassen(5, [(2,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(3,0,1)]) # C22
    # M7
    set_strassen(6, [(0,1,1), (3,0,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (2,0,-1)]) # C11, C21

    # 3. Kronecker Product M_1,2,2 (R=2) and M_2,2,2 (R=7) to get M_2,4,4 (R=14)
    # n=1*2=2, m=2*2=4, p=2*2=4
    n12, m12, p12 = 2, 4, 4
    R12 = R1 * R2
    U12 = [[0]*R12 for _ in range(n12*m12)]
    V12 = [[0]*R12 for _ in range(m12*p12)]
    W12 = [[0]*R12 for _ in range(p12*n12)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    idx_u = i * m12 + j
                    for r1 in range(R1):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = U1[i1*m1+j1][r1] * U2[i2*m2+j2][r2]
                            U12[idx_u][r] = val
                            
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    idx_v = j * p12 + k
                    for r1 in range(R1):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = V1[j1*p1+k1][r1] * V2[j2*p2+k2][r2]
                            V12[idx_v][r] = val
                            
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    idx_w = k * n12 + i
                    for r1 in range(R1):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = W1[k1*n1+i1][r1] * W2[k2*n2+i2][r2]
                            W12[idx_w][r] = val

    # 4. Kronecker Product M_2,4,4 (R=14) and M_2,2,2 (R=7) to get M_4,8,8 (R=98)
    n22, m22, p22 = 4, 8, 8
    R22 = R12 * R2
    U22 = [[0]*R22 for _ in range(n22*m22)]
    V22 = [[0]*R22 for _ in range(m22*p22)]
    W22 = [[0]*R22 for _ in range(p22*n22)]
    
    # M_2,4,4 is T1, M_2,2,2 is T2
    # n=2*2=4, m=4*2=8, p=4*2=8
    for i1 in range(n12):
        for j1 in range(m12):
            for i2 in range(n2):
                for j2 in range(m2):
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    idx_u = i * m22 + j
                    for r1 in range(R12):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = U12[i1*m12+j1][r1] * U2[i2*m2+j2][r2]
                            U22[idx_u][r] = val
                            
    for j1 in range(m12):
        for k1 in range(p12):
            for j2 in range(m2):
                for k2 in range(p2):
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    idx_v = j * p22 + k
                    for r1 in range(R12):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = V12[j1*p12+k1][r1] * V2[j2*p2+k2][r2]
                            V22[idx_v][r] = val
                            
    for k1 in range(p12):
        for i1 in range(n12):
            for k2 in range(p2):
                for i2 in range(n2):
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    idx_w = k * n22 + i
                    for r1 in range(R12):
                        for r2 in range(R2):
                            r = r1 * R2 + r2
                            val = W12[k1*n12+i1][r1] * W2[k2*n2+i2][r2]
                            W22[idx_w][r] = val

    # 5. Restrict to 3x5x7
    n, m, p = 3, 5, 7
    U = [[0]*R22 for _ in range(n*m)]
    V = [[0]*R22 for _ in range(m*p)]
    W = [[0]*R22 for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            # Map to 4x8 index: i_4 = i, j_4 = j
            idx_u_22 = i * 8 + j
            for r in range(R22):
                U[idx_u][r] = U22[idx_u_22][r]
                
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            # Map to 8x8 index: j_8 = j, k_8 = k
            idx_v_22 = j * 8 + k
            for r in range(R22):
                V[idx_v][r] = V22[idx_v_22][r]
                
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            # Map to 8x4 index: k_8 = k, i_4 = i
            idx_w_22 = k * 4 + i
            for r in range(R22):
                W[idx_w][r] = W22[idx_w_22][r]
                
    return {"U": U, "V": V, "W": W}
