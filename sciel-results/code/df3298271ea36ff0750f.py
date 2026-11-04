import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    # U2, V2, W2 of shape (4, 7)
    # Indices: i, j in {0,1} -> row = i*2 + j
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # c00, c11
    
    # M2 = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # c10, c11
    
    # M3 = a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # c01, c11
    
    # M4 = a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # c00, c10
    
    # M5 = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # c11
    
    # M7 = (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # c00
    
    # Verify Strassen
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Construct 4x4x4
    # U4, V4, W4 of shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            ib = i // 2
            io = i % 2
            jb = j // 2
            jo = j % 2
            row_u = i * 4 + j
            row_v = j * 4 + i # Wait, V is j, k. For 4x4x4, k is also 0..3.
            # V index is j * p + k. Here p=4. So row_v = j * 4 + k.
            # But we are constructing V4 for 4x4x4, so k is 0..3.
            # W index is k * n + i. Here n=4. So row_w = k * 4 + i.
            
            # U4[row_u, r] = U2[ib*2+jb, rb] * U2[io*2+jo, ro]
            # r = rb * 7 + ro
            for rb in range(7):
                for ro in range(7):
                    r = rb * 7 + ro
                    U4[row_u, r] = U2[ib*2+jb, rb] * U2[io*2+jo, ro]
                    
                    # V4[row_v, r] = V2[jb*2+kb, rb] * V2[jo*2+ko, ro]
                    # But V4 depends on j, k.
                    # We need to fill V4 for all k.
                    # So we should loop over k as well.
                    
            # Let's restructure the loop.
            
    # Re-do 4x4x4 construction
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        ib = i // 2
        io = i % 2
        for j in range(4):
            jb = j // 2
            jo = j % 2
            row_u = i * 4 + j
            for k in range(4):
                kb = k // 2
                ko = k % 2
                row_v = j * 4 + k
                row_w = k * 4 + i
                
                for rb in range(7):
                    for ro in range(7):
                        r = rb * 7 + ro
                        U4[row_u, r] += U2[ib*2+jb, rb] * U2[io*2+jo, ro]
                        V4[row_v, r] += V2[jb*2+kb, rb] * V2[jo*2+ko, ro]
                        W4[row_w, r] += W2[kb*2+ib, rb] * W2[ko*2+io, ro]
                        
    # Wait, U4 should not depend on k.
    # U4[row_u, r] is the same for all k.
    # So we can just compute it once.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        ib = i // 2
        io = i % 2
        for j in range(4):
            jb = j // 2
            jo = j % 2
            row_u = i * 4 + j
            for rb in range(7):
                for ro in range(7):
                    r = rb * 7 + ro
                    U4[row_u, r] = U2[ib*2+jb, rb] * U2[io*2+jo, ro]
                    
    for j in range(4):
        jb = j // 2
        jo = j % 2
        for k in range(4):
            kb = k // 2
            ko = k % 2
            row_v = j * 4 + k
            for rb in range(7):
                for ro in range(7):
                    r = rb * 7 + ro
                    V4[row_v, r] = V2[jb*2+kb, rb] * V2[jo*2+ko, ro]
                    
    for k in range(4):
        kb = k // 2
        ko = k % 2
        for i in range(4):
            ib = i // 2
            io = i % 2
            row_w = k * 4 + i
            for rb in range(7):
                for ro in range(7):
                    r = rb * 7 + ro
                    W4[row_w, r] = W2[kb*2+ib, rb] * W2[ko*2+io, ro]
                    
    # Now construct 4x4x9
    # Rank 114
    U = np.zeros((16, 114), dtype=int)
    V = np.zeros((36, 114), dtype=int)
    W = np.zeros((36, 114), dtype=int)
    
    # Part 1: k=0..3 (B1)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V4[j*4+k]
            W[k*4+i, 0:49] = W4[k*4+i] # Wait, W depends on i too.
            
    # W4 is 16x49. W is 36x114.
    # W row index is k*n + i = k*4 + i.
    # So W[k*4+i, 0:49] = W4[k*4+i]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i]
            
    # Part 2: k=4..7 (B2)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+(k+4), 49:98] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i]
            
    # Part 3: k=8 (Tail)
    for i in range(4):
        for j in range(4):
            r = 98 + i*4 + j
            U[i*4+j, r] = 1
            V[j*9+8, r] = 1
            W[8*4+i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
