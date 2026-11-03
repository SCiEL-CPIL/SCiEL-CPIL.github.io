import numpy as np

def construct():
    # Strassen for 2x2x2
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    # Indices:
    # U1 row: i*2 + j
    # V1 row: j*2 + k
    # W1 row: k*2 + i
    
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # P0 = (a+d)(e+h)
    # A: a=0,0 (idx 0), d=1,1 (idx 3)
    # B: e=0,0 (idx 0), h=1,1 (idx 3)
    # C: c00=0,0 (idx 0), c11=1,1 (idx 3) -> W idx k*2+i
    # c00: k=0, i=0 -> 0
    # c11: k=1, i=1 -> 1*2+1=3
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # P1 = (c+d)e
    # A: c=1,0 (idx 2), d=1,1 (idx 3)
    # B: e=0,0 (idx 0)
    # C: c10=1,0 (idx 2), c11=1,1 (idx 3)
    # c10: k=0, i=1 -> 0*2+1=1
    # c11: k=1, i=1 -> 1*2+1=3
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # P2 = a(f-h)
    # A: a=0,0 (idx 0)
    # B: f=0,1 (idx 1), h=1,1 (idx 3)
    # C: c01=0,1 (idx 1), c11=1,1 (idx 3)
    # c01: k=1, i=0 -> 1*2+0=2
    # c11: k=1, i=1 -> 1*2+1=3
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # P3 = d(g-e)
    # A: d=1,1 (idx 3)
    # B: g=1,0 (idx 2), e=0,0 (idx 0)
    # C: c00=0,0 (idx 0), c10=1,0 (idx 2)
    # c00: k=0, i=0 -> 0
    # c10: k=0, i=1 -> 1
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # P4 = (a+b)h
    # A: a=0,0 (idx 0), b=0,1 (idx 1)
    # B: h=1,1 (idx 3)
    # C: c00=0,0 (idx 0), c01=0,1 (idx 2)
    # c00: k=0, i=0 -> 0
    # c01: k=1, i=0 -> 2
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # P5 = (c-a)(e+f)
    # A: c=1,0 (idx 2), a=0,0 (idx 0)
    # B: e=0,0 (idx 0), f=0,1 (idx 1)
    # C: c11=1,1 (idx 3)
    # c11: k=1, i=1 -> 3
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # P6 = (b-d)(g+h)
    # A: b=0,1 (idx 1), d=1,1 (idx 3)
    # B: g=1,0 (idx 2), h=1,1 (idx 3)
    # C: c00=0,0 (idx 0)
    # c00: k=0, i=0 -> 0
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    # Construct U2, V2, W2 for 2x2x4
    # U2: (4, 14)
    # V2: (8, 14)
    # W2: (8, 14)
    
    U2 = np.zeros((4, 14), dtype=int)
    V2 = np.zeros((8, 14), dtype=int)
    W2 = np.zeros((8, 14), dtype=int)
    
    # First 7 products: A * B[:, 0:2] -> C[:, 0:2]
    # U2[:, 0:7] = U1
    U2[:, 0:7] = U1
    # V2 rows 0:4 correspond to B[:, 0:2]
    # V1 indices are j*2+k. j in 0,1, k in 0,1.
    # B[:, 0:2] has cols 0,1. So V2[0:4, 0:7] = V1
    V2[0:4, 0:7] = V1
    # W2 rows 0:4 correspond to C[:, 0:2] (k=0,1)
    # W1 indices are k*2+i. k in 0,1.
    # So W2[0:4, 0:7] = W1
    W2[0:4, 0:7] = W1
    
    # Next 7 products: A * B[:, 2:4] -> C[:, 2:4]
    # U2[:, 7:14] = U1
    U2[:, 7:14] = U1
    # V2 rows 4:8 correspond to B[:, 2:4]
    # V1 indices j*2+k. Here B cols are 2,3.
    # So we map V1 index (j*2+k) to V2 index (j*4 + (2+k))?
    # V2 row index is j*4 + k_total.
    # For B[:, 2:4], k_total is 2 or 3.
    # So V2[j*4 + (2+k), 7:14] = V1[j*2+k, :]
    for j in range(2):
        for k in range(2):
            v2_idx = j * 4 + (2 + k)
            v1_idx = j * 2 + k
            V2[v2_idx, 7:14] = V1[v1_idx, :]
            
    # W2 rows 4:8 correspond to C[:, 2:4] (k=2,3)
    # W1 indices k*2+i. Here C cols are 2,3.
    # So W2[(2+k)*2 + i, 7:14] = W1[k*2+i, :]
    for k in range(2):
        for i in range(2):
            w2_idx = (2 + k) * 2 + i
            w1_idx = k * 2 + i
            W2[w2_idx, 7:14] = W1[w1_idx, :]
            
    # Now Kronecker product for 4x4x8
    # n1=2, m1=2, p1=2
    # n2=2, m2=2, p2=4
    # n=4, m=4, p=8
    
    # U: (16, 98)
    # U_total[i1*4 + i2, r1*14 + r2] = U1[i1*2+j1, r1] * U2[i2*2+j2, r2] ?
    # No, U1 is (4,7). Rows are i1*2+j1.
    # U2 is (4,14). Rows are i2*2+j2.
    # Total U row index: (i1*2 + i2) * 4 + (j1*2 + j2) ?
    # Let's check: A is 4x4. i in 0..3, j in 0..3.
    # i = i1*2 + i2. j = j1*2 + j2.
    # Lin idx = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    
    # Rank 98 = 7 * 14. r = r1*14 + r2.
    
    U_total = np.zeros((16, 98), dtype=int)
    V_total = np.zeros((32, 98), dtype=int)
    W_total = np.zeros((32, 98), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    u_idx = (i1*2 + i2)*4 + (j1*2 + j2)
                    u1_idx = i1*2 + j1
                    u2_idx = i2*2 + j2
                    for r1 in range(7):
                        for r2 in range(14):
                            r_idx = r1*14 + r2
                            U_total[u_idx, r_idx] = U1[u1_idx, r1] * U2[u2_idx, r2]
                            
    # V: (32, 98)
    # V_total row index: j*8 + k.
    # j = j1*2 + j2. k = k1*4 + k2.
    # Lin idx = (j1*2+j2)*8 + (k1*4+k2) = 16*j1 + 8*j2 + 4*k1 + k2.
    # V1 idx: j1*2 + k1.
    # V2 idx: j2*4 + k2.
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(4):
                    v_idx = (j1*2 + j2)*8 + (k1*4 + k2)
                    v1_idx = j1*2 + k1
                    v2_idx = j2*4 + k2
                    for r1 in range(7):
                        for r2 in range(14):
                            r_idx = r1*14 + r2
                            V_total[v_idx, r_idx] = V1[v1_idx, r1] * V2[v2_idx, r2]
                            
    # W: (32, 98)
    # W_total row index: k*n + i = k*4 + i.
    # k = k1*4 + k2. i = i1*2 + i2.
    # Lin idx = (k1*4+k2)*4 + (i1*2+i2) = 16*k1 + 4*k2 + 2*i1 + i2.
    # W1 idx: k1*2 + i1.
    # W2 idx: k2*2 + i2.
    
    for k1 in range(2):
        for k2 in range(4):
            for i1 in range(2):
                for i2 in range(2):
                    w_idx = (k1*4 + k2)*4 + (i1*2 + i2)
                    w1_idx = k1*2 + i1
                    w2_idx = k2*2 + i2
                    for r1 in range(7):
                        for r2 in range(14):
                            r_idx = r1*14 + r2
                            W_total[w_idx, r_idx] = W1[w1_idx, r1] * W2[w2_idx, r2]
                            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
