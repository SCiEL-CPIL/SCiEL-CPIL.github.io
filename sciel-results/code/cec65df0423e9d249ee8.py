import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Start with time check
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U, V, W are 4x7
    U222 = np.zeros((4, 7), dtype=int)
    V222 = np.zeros((4, 7), dtype=int)
    W222 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U222[0, 0] = 1; U222[3, 0] = 1
    V222[0, 0] = 1; V222[3, 0] = 1
    W222[0, 0] = 1; W222[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U222[2, 1] = 1; U222[3, 1] = 1
    V222[0, 1] = 1
    W222[2, 1] = 1
    
    # M3: A00(B10-B11) -> C01
    U222[0, 2] = 1
    V222[2, 2] = 1; V222[3, 2] = -1
    W222[1, 2] = 1
    
    # M4: A11(B01+B11) -> C11
    U222[3, 3] = 1
    V222[1, 3] = 1; V222[3, 3] = 1
    W222[3, 3] = 1
    
    # M5: (A00+A01)B11 -> C00
    U222[0, 4] = 1; U222[1, 4] = 1
    V222[3, 4] = 1
    W222[0, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C10
    U222[2, 5] = 1; U222[0, 5] = -1
    V222[0, 5] = 1; V222[1, 5] = 1
    W222[2, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C01
    U222[1, 6] = 1; U222[3, 6] = -1
    V222[2, 6] = 1; V222[3, 6] = 1
    W222[1, 6] = 1
    
    # Adjust W for C00 and C01 to match Strassen formulas
    # C00 = M1 + M5 - M4 - M7
    # C01 = M4 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # My W222 currently has:
    # W[0] (C00): M1(1), M5(1). Missing -M4, -M7.
    # W[1] (C01): M3(1), M7(1). Missing M4.
    # W[2] (C10): M2(1), M6(1). Correct.
    # W[3] (C11): M1(1), M4(1). Missing -M2, M3.
    
    W222[0, 3] -= 1
    W222[0, 6] -= 1
    W222[1, 3] += 1
    W222[3, 1] -= 1
    W222[3, 2] += 1
    
    # 2. Construct 4x4x4 via Kronecker product
    # U444 = U222 (x) U222
    U444 = np.kron(U222, U222)
    V444 = np.kron(V222, V222)
    W444 = np.kron(W222, W222)
    
    # 3. Construct 4x4x1 block (for C_i4 = sum A_ij B_j4)
    # Rank 16
    U441 = np.zeros((16, 16), dtype=int)
    V441 = np.zeros((16, 16), dtype=int)
    W441 = np.zeros((4, 16), dtype=int)
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U441[i * 4 + j, r] = 1
            V441[j, r] = 1 # V index for B_j4 is just j in local 4x1? No, V is 4x1, index j.
            W441[i, r] = 1
    
    # 4. Construct 4x1x5 block (for C_ik += A_i4 B_4k)
    # Rank 20
    U415 = np.zeros((4, 20), dtype=int)
    V415 = np.zeros((5, 20), dtype=int)
    W415 = np.zeros((20, 20), dtype=int)
    
    for i in range(4):
        for k in range(5):
            r = i * 5 + k
            U415[i, r] = 1
            V415[k, r] = 1
            W415[k * 4 + i, r] = 1
    
    # 5. Combine into global 4x5x5
    # Total rank = 49 + 16 + 20 = 85
    R = 85
    U = np.zeros((20, R), dtype=int)
    V = np.zeros((25, R), dtype=int)
    W = np.zeros((20, R), dtype=int)
    
    # Block 1: 4x4x4
    # Local indices i,j,k in 0..3
    # Global u = i*5 + j
    # Global v = j*5 + k
    # Global w = k*4 + i
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = i * 5 + j
                v = j * 5 + k
                w = k * 4 + i
                
                # Map local 4x4x4 indices to U444, V444, W444
                # U444 row is i*4+j
                # V444 row is j*4+k
                # W444 row is k*4+i
                
                u_loc = i * 4 + j
                v_loc = j * 4 + k
                w_loc = k * 4 + i
                
                U[u, 0:49] += U444[u_loc, :]
                V[v, 0:49] += V444[v_loc, :]
                W[w, 0:49] += W444[w_loc, :]
                
    # Block 2: 4x4x1 (C_i4)
    # i in 0..3, j in 0..3
    # u = i*5 + j
    # v = j*5 + 4
    # w = 4*4 + i
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            u = i * 5 + j
            v = j * 5 + 4
            w = 4 * 4 + i
            
            U[u, 49 + r] = 1
            V[v, 49 + r] = 1
            W[w, 49 + r] = 1
            
    # Block 3: 4x1x5 (A_i4 B_4k)
    # i in 0..3, k in 0..4
    # u = i*5 + 4
    # v = 4*5 + k
    # w = k*4 + i
    
    for i in range(4):
        for k in range(5):
            r = i * 5 + k
            u = i * 5 + 4
            v = 4 * 5 + k
            w = k * 4 + i
            
            U[u, 65 + r] = 1
            V[v, 65 + r] = 1
            W[w, 65 + r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

