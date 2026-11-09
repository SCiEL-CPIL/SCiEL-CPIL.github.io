import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Returns U, V, W for n x m x p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need to map these to the tensor indices.
    # U indices: i*2+j. (0:00, 1:01, 2:10, 3:11)
    # V indices: j*2+k. (0:00, 1:01, 2:10, 3:11)
    # W indices: k*2+i. (0:00, 1:10, 2:01, 3:11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    # a=A00, d=A11, e=B00, h=B11
    # U: A00, A11 -> U[0,0]=1, U[3,0]=1
    # V: B00, B11 -> V[0,0]=1, V[3,0]=1
    # W: C00, C11, C01, C10 contributions?
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Let's just populate the factors directly based on the linear combinations.
    # This is tedious to do manually for all 7.
    # Alternative: Use the known explicit factors.
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = 1; W[2, 0] = 1 # k*n+i
    # W indices:
    # C00 (i=0, k=0) -> W[0]
    # C11 (i=1, k=1) -> W[3]
    # C01 (i=0, k=1) -> W[2]
    # C10 (i=1, k=0) -> W[1]
    # M1 contributes to all 4.
    
    # M2: (c+d)e
    # c=A10, d=A11, e=B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10 = M2+M4, C11 = M1-M2...
    # Wait, C10 = M2 + M4. C11 = M1 - M2 - M3 + M6.
    # So M2 contributes +1 to C10 (W[1]) and -1 to C11 (W[3]).
    
    # M3: a(f-h)
    # a=A00, f=B01, h=B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1 # C01 = M3+M5, C11 = ... -M3...
    # C01 (W[2]) gets +1. C11 (W[3]) gets -1.
    
    # M4: d(g-e)
    # d=A11, g=B10, e=B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = -1 # C10 = M2+M4, C00 = M1-M4...
    # C10 (W[1]) gets +1. C00 (W[0]) gets -1.
    
    # M5: (a+b)h
    # a=A00, b=A01, h=B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00 = ... -M5, C01 = M3+M5
    # C00 (W[0]) gets -1. C01 (W[2]) gets +1.
    
    # M6: (c-a)(e+f)
    # c=A10, a=A00, e=B00, f=B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 = ... +M6
    # C11 (W[3]) gets +1.
    
    # M7: (b-d)(g+h)
    # b=A01, d=A11, g=B10, h=B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 = ... +M7
    # C00 (W[0]) gets +1.
    
    return U, V, W

def _strassen_4x4():
    """Returns U, V, W for 4x4x4 Strassen decomposition."""
    # Block Strassen on 4x4 matrices using 2x2 blocks.
    # 7 products of 2x2 matrices.
    U_2, V_2, W_2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Blocks:
    # A: 00, 01, 10, 11 (each 2x2)
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will embed the 2x2 decomposition into each block product.
    
    def embed(U_sub, V_sub, W_sub, a_offset, b_offset, c_offset, col_offset):
        # a_offset: (row_block, col_block) of A
        # b_offset: (row_block, col_block) of B
        # c_offset: (row_block, col_block) of C
        # U_sub is 4x7. U is 16x49.
        # U row in sub: i*2+j. Global: (a_rb*2+i)*4 + (a_cb*2+j)
        # V row in sub: j*2+k. Global: (b_rb*2+j)*4 + (b_cb*2+k)
        # W row in sub: k*2+i. Global: (c_cb*2+k)*4 + (c_rb*2+i)
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_sub_idx = i * 2 + j
                    u_glob_idx = (a_offset[0] * 2 + i) * 4 + (a_offset[1] * 2 + j)
                    U[u_glob_idx, col_offset + r] += U_sub[u_sub_idx, r]
                    
                    v_sub_idx = j * 2 + i # Wait, V index is j*2+k.
                    # In 2x2 decomp, V index is j*2+k.
                    # Here j is local col of A, which is local row of B.
                    # So v_sub_idx = j * 2 + k.
                    # We need to iterate k too?
                    # No, U_sub, V_sub, W_sub are already fully populated.
                    # We just map the indices.
                    
            # Let's just do a full index mapping loop.
            for u_idx in range(4):
                i, j = divmod(u_idx, 2)
                u_glob = (a_offset[0] * 2 + i) * 4 + (a_offset[1] * 2 + j)
                U[u_glob, col_offset:col_offset+7] += U_sub[u_idx, :]
            
            for v_idx in range(4):
                j, k = divmod(v_idx, 2)
                v_glob = (b_offset[0] * 2 + j) * 4 + (b_offset[1] * 2 + k)
                V[v_glob, col_offset:col_offset+7] += V_sub[v_idx, :]
            
            for w_idx in range(4):
                k, i = divmod(w_idx, 2)
                w_glob = (c_offset[1] * 2 + k) * 4 + (c_offset[0] * 2 + i)
                W[w_glob, col_offset:col_offset+7] += W_sub[w_idx, :]

    # M1
    # A: 00+11, B: 00+11
    # We need to add two 2x2 decompositions?
    # No, Strassen factors are linear.
    # U for (A00+A11) is U00 + U11.
    # So we call embed for A00 and A11, adding to the same columns.
    
    col = 0
    # M1
    embed(U_2, V_2, W_2, (0,0), (0,0), (0,0), col)
    embed(U_2, V_2, W_2, (1,1), (1,1), (0,0), col)
    col += 7
    
    # M2
    # A: 10+11, B: 00
    embed(U_2, V_2, W_2, (1,0), (0,0), (1,0), col)
    embed(U_2, V_2, W_2, (1,1), (0,0), (1,0), col)
    col += 7
    
    # M3
    # A: 00, B: 01-11
    embed(U_2, V_2, W_2, (0,0), (0,1), (0,1), col)
    embed(-U_2, -V_2, -W_2, (0,0), (1,1), (0,1), col)
    col += 7
    
    # M4
    # A: 11, B: 10-00
    embed(U_2, V_2, W_2, (1,1), (1,0), (1,1), col)
    embed(-U_2, -V_2, -W_2, (1,1), (0,0), (1,1), col)
    col += 7
    
    # M5
    # A: 00+01, B: 11
    embed(U_2, V_2, W_2, (0,0), (1,1), (0,1), col)
    embed(U_2, V_2, W_2, (0,1), (1,1), (0,1), col)
    col += 7
    
    # M6
    # A: 10-00, B: 00+01
    embed(U_2, V_2, W_2, (1,0), (0,0), (1,1), col)
    embed(-U_2, -V_2, -W_2, (0,0), (0,0), (1,1), col)
    embed(U_2, V_2, W_2, (1,0), (0,1), (1,1), col)
    embed(-U_2, -V_2, -W_2, (0,0), (0,1), (1,1), col)
    col += 7
    
    # M7
    # A: 01-11, B: 10+11
    embed(U_2, V_2, W_2, (0,1), (1,0), (0,0), col)
    embed(-U_2, -V_2, -W_2, (1,1), (1,0), (0,0), col)
    embed(U_2, V_2, W_2, (0,1), (1,1), (0,0), col)
    embed(-U_2, -V_2, -W_2, (1,1), (1,1), (0,0), col)
    col += 7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Sub-problems for 5x5 using 4+1 split
    # 1. A44 B44 (4x4x4)
    U1, V1, W1 = _strassen_4x4()
    
    # 2. A41 B14 (4x1x4)
    U2, V2, W2 = _schoolbook(4, 1, 4)
    
    # 3. A44 B41 (4x4x1)
    U3, V3, W3 = _schoolbook(4, 4, 1)
    
    # 4. A41 B11 (4x1x1)
    U4, V4, W4 = _schoolbook(4, 1, 1)
    
    # 5. A14 B44 (1x4x4)
    U5, V5, W5 = _schoolbook(1, 4, 4)
    
    # 6. A11 B14 (1x1x4)
    U6, V6, W6 = _schoolbook(1, 1, 4)
    
    # 7. A14 B41 (1x4x1)
    U7, V7, W7 = _schoolbook(1, 4, 1)
    
    # 8. A11 B11 (1x1x1)
    U8, V8, W8 = _schoolbook(1, 1, 1)
    
    # Combine
    rank = U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1] + U5.shape[1] + U6.shape[1] + U7.shape[1] + U8.shape[1]
    
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    col = 0
    
    # 1. A44 B44
    # A: i in 0..3, k in 0..3. U idx i*5+k
    # B: k in 0..3, j in 0..3. V idx k*5+j
    # C: i in 0..3, j in 0..3. W idx j*5+i
    for i in range(4):
        for k in range(4):
            u_sub = i * 4 + k
            u_glob = i * 5 + k
            U[u_glob, col:col+49] += U1[u_sub, :]
    for k in range(4):
        for j in range(4):
            v_sub = k * 4 + j
            v_glob = k * 5 + j
            V[v_glob, col:col+49] += V1[v_sub, :]
    for j in range(4):
        for i in range(4):
            w_sub = j * 4 + i
            w_glob = j * 5 + i
            W[w_glob, col:col+49] += W1[w_sub, :]
    col += 49
    
    # 2. A41 B14
    # A: i in 0..3, k=4. U idx i*5+4
    # B: k=4, j in 0..3. V idx 4*5+j
    # C: i in 0..3, j in 0..3. W idx j*5+i
    for i in range(4):
        u_sub = i * 1 + 0
        u_glob = i * 5 + 4
        U[u_glob, col:col+16] += U2[u_sub, :]
    for j in range(4):
        v_sub = 0 * 4 + j
        v_glob = 4 * 5 + j
        V[v_glob, col:col+16] += V2[v_sub, :]
    for j in range(4):
        for i in range(4):
            w_sub = 0 * 4 + i
            w_glob = j * 5 + i
            W[w_glob, col:col+16] += W2[w_sub, :]
    col += 16
    
    # 3. A44 B41
    # A: i in 0..3, k in 0..3. U idx i*5+k
    # B: k in 0..3, j=4. V idx k*5+4
    # C: i in 0..3, j=4. W idx 4*5+i
    for i in range(4):
        for k in range(4):
            u_sub = i * 4 + k
            u_glob = i * 5 + k
            U[u_glob, col:col+16] += U3[u_sub, :]
    for k in range(4):
        v_sub = k * 1 + 0
        v_glob = k * 5 + 4
        V[v_glob, col:col+16] += V3[v_sub, :]
    for i in range(4):
        w_sub = 0 * 4 + i
        w_glob = 4 * 5 + i
        W[w_glob, col:col+16] += W3[w_sub, :]
    col += 16
    
    # 4. A41 B11
    # A: i in 0..3, k=4. U idx i*5+4
    # B: k=4, j=4. V idx 4*5+4
    # C: i in 0..3, j=4. W idx 4*5+i
    for i in range(4):
        u_sub = i
        u_glob = i * 5 + 4
        U[u_glob, col:col+4] += U4[u_sub, :]
    v_sub = 0
    v_glob = 4 * 5 + 4
    V[v_glob, col:col+4] += V2[0, :] # Wait, V4
    V[v_glob, col:col+4] += V4[0, :]
    for i in range(4):
        w_sub = i
        w_glob = 4 * 5 + i
        W[w_glob, col:col+4] += W4[w_sub, :]
    col += 4
    
    # 5. A14 B44
    # A: i=4, k in 0..3. U idx 4*5+k
    # B: k in 0..3, j in 0..3. V idx k*5+j
    # C: i=4, j in 0..3. W idx j*5+4
    for k in range(4):
        u_sub = 0 * 4 + k
        u_glob = 4 * 5 + k
        U[u_glob, col:col+16] += U5[u_sub, :]
    for k in range(4):
        for j in range(4):
            v_sub = k * 4 + j
            v_glob = k * 5 + j
            V[v_glob, col:col+16] += V5[v_sub, :]
    for j in range(4):
        w_sub = j * 1 + 0
        w_glob = j * 5 + 4
        W[w_glob, col:col+16] += W5[w_sub, :]
    col += 16
    
    # 6. A11 B14
    # A: i=4, k=4. U idx 4*5+4
    # B: k=4, j in 0..3. V idx 4*5+j
    # C: i=4, j in 0..3. W idx j*5+4
    u_sub = 0
    u_glob = 4 * 5 + 4
    U[u_glob, col:col+4] += U6[0, :]
    for j in range(4):
        v_sub = 0 * 4 + j
        v_glob = 4 * 5 + j
        V[v_glob, col:col+4] += V6[v_sub, :]
    for j in range(4):
        w_sub = j
        w_glob = j * 5 + 4
        W[w_glob, col:col+4] += W6[w_sub, :]
    col += 4
    
    # 7. A14 B41
    # A: i=4, k in 0..3. U idx 4*5+k
    # B: k in 0..3, j=4. V idx k*5+4
    # C: i=4, j=4. W idx 4*5+4
    for k in range(4):
        u_sub = 0 * 4 + k
        u_glob = 4 * 5 + k
        U[u_glob, col:col+4] += U7[u_sub, :]
    for k in range(4):
        v_sub = k
        v_glob = k * 5 + 4
        V[v_glob, col:col+4] += V7[v_sub, :]
    w_sub = 0
    w_glob = 4 * 5 + 4
    W[w_glob, col:col+4] += W7[0, :]
    col += 4
    
    # 8. A11 B11
    # A: i=4, k=4. U idx 4*5+4
    # B: k=4, j=4. V idx 4*5+4
    # C: i=4, j=4. W idx 4*5+4
    u_sub = 0
    u_glob = 4 * 5 + 4
    U[u_glob, col:col+1] += U8[0, :]
    v_sub = 0
    v_glob = 4 * 5 + 4
    V[v_glob, col:col+1] += V8[0, :]
    w_sub = 0
    w_glob = 4 * 5 + 4
    W[w_glob, col:col+1] += W8[0, :]
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
