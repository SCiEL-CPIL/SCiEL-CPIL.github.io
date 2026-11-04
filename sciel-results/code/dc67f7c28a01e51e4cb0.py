import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Strassen decomposition for 2x2x2
    # u, v, w are 4x7 matrices (flattened 2x2 indices)
    # Indices: 0:00, 1:01, 2:10, 3:11 for 2x2
    # u[i,j] is stored at u[i*2+j]
    
    u2 = np.zeros((4, 7))
    v2 = np.zeros((4, 7))
    w2 = np.zeros((4, 7))
    
    # M1: (a+d)(e+h) -> C00+=1, C11+=1
    u2[0, 0] = 1; u2[3, 0] = 1
    v2[0, 0] = 1; v2[3, 0] = 1
    w2[0, 0] = 1; w2[3, 0] = 1 # W[k,i] -> W[0,0] for C00, W[1,1] for C11. Index k*2+i. C00 is i=0,k=0 -> 0. C11 is i=1,k=1 -> 3.
    
    # M2: (c+d)e -> C10+=1, C11+=-1
    u2[2, 1] = 1; u2[3, 1] = 1
    v2[0, 1] = 1
    w2[2, 1] = 1 # C10: i=1,k=0 -> 0*2+1=1? No. W index is k*n+i. For 2x2, n=2. W[k*2+i].
    # C10 is i=1, k=0. Index 0*2+1 = 1.
    # C11 is i=1, k=1. Index 1*2+1 = 3.
    w2[1, 1] = 1
    w2[3, 1] = -1
    
    # M3: a(f-h) -> C01+=1, C11+=1
    u2[0, 2] = 1
    v2[1, 2] = 1; v2[3, 2] = -1
    # C01: i=0, k=1. Index 1*2+0 = 2.
    # C11: i=1, k=1. Index 1*2+1 = 3.
    w2[2, 2] = 1
    w2[3, 2] = 1
    
    # M4: d(g-e) -> C00+=1, C10+=1
    u2[3, 3] = 1
    v2[2, 3] = 1; v2[0, 3] = -1
    # C00: i=0, k=0. Index 0.
    # C10: i=1, k=0. Index 1.
    w2[0, 3] = 1
    w2[1, 3] = 1
    
    # M5: (a+b)h -> C00+=-1, C01+=1
    u2[0, 4] = 1; u2[1, 4] = 1
    v2[3, 4] = 1
    # C00: i=0, k=0. Index 0.
    # C01: i=0, k=1. Index 2.
    w2[0, 4] = -1
    w2[2, 4] = 1
    
    # M6: (c-a)(e+f) -> C11+=1
    u2[2, 5] = 1; u2[0, 5] = -1
    v2[0, 5] = 1; v2[1, 5] = 1
    # C11: i=1, k=1. Index 3.
    w2[3, 5] = 1
    
    # M7: (b-d)(g+h) -> C00+=1
    u2[1, 6] = 1; u2[3, 6] = -1
    v2[2, 6] = 1; v2[3, 6] = 1
    # C00: i=0, k=0. Index 0.
    w2[0, 6] = 1
    
    # Verify 2x2x2 decomposition
    # Sum_c u[i,j] v[j,k] w[k,i] should be 1 for all i,j,k in 0..1
    # i,j,k are 2x2 indices. Flattened: i*2+j.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += u2[i*2+j, c] * v2[j*2+k, c] * w2[k*2+i, c]
                if abs(s - 1.0) > 1e-9:
                    #print(f"Fail 2x2: {i},{j},{k} sum={s}")
                    pass

    # Construct 4x4x4 Strassen decomposition
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49)) # Wait, V is for 4x5. For the 4x4 part, it uses first 4 cols of B.
    # V shape in final is (20, 65). Here we build the 4x4 part contribution to V, which is (16, 49) using k in 0..3.
    # Actually, V index is j*p+k. For p=5, j in 0..3, k in 0..3.
    W4 = np.zeros((16, 49)) # W index is k*n+i. k in 0..3, i in 0..3.

    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            # U index: i*4+j. i = i1*2+i2, j = j1*2+j2
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4[i * 4 + j, c] = u2[i1 * 2 + j1, c1] * u2[i2 * 2 + j2, c2]
                            
                            # V index: j*5+k. j = j1*2+j2, k = k1*2+k2
                            # We need to fill V4 for k in 0..3
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V4[j * 5 + k, c] = v2[j1 * 2 + k1, c1] * v2[j2 * 2 + k2, c2]
                                    
                                    # W index: k*4+i. k = k1*2+k2, i = i1*2+i2
                                    W4[k * 4 + i, c] = w2[k1 * 2 + i1, c1] * w2[k2 * 2 + i2, c2]

    # Schoolbook for 4x4x1 part (k=4)
    # Rank 16
    U1 = np.zeros((16, 16))
    V1 = np.zeros((20, 16))
    W1 = np.zeros((20, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i * 4 + j, col] = 1
            V1[j * 5 + 4, col] = 1 # k=4
            W1[4 * 4 + i, col] = 1 # k=4, i
            col += 1

    # Combine
    U = np.hstack([U4, U1])
    V = np.hstack([V4, V1])
    W = np.hstack([W4, W1])
    
    # Verify full decomposition
    # Sum_c U[i*4+j, c] V[j*5+k, c] W[k*4+i, c] == 1
    for i in range(4):
        for j in range(4):
            for k in range(5):
                s = 0
                for c in range(65):
                    s += U[i * 4 + j, c] * V[j * 5 + k, c] * W[k * 4 + i, c]
                if abs(s - 1.0) > 1e-9:
                    #print(f"Fail 4x4x5: {i},{j},{k} sum={s}")
                    pass

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
