import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00(+), C11(+)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10(+), C11(+)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11) -> C01(+), C11(-)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4: A11(B10-B00) -> C10(+), C00(-)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[0, 3] = -1
    
    # M5: (A00+A01)B11 -> C01(+), C11(-)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1
    
    # M6: (A10-A00)(B00+B01) -> C00(+), C10(-)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1; W2[2, 5] = -1
    
    # M7: (A01-A11)(B10+B11) -> C00(-), C11(-)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1

    # 2. Construct 4x4x4 decomposition using Kronecker product
    # Rank 49
    U444 = np.zeros((16, 49), dtype=float)
    V444 = np.zeros((16, 49), dtype=float)
    W444 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        i1, i0 = divmod(i, 2)
        for j in range(4):
            j1, j0 = divmod(j, 2)
            idx_u = i * 4 + j
            
            for k in range(4):
                k1, k0 = divmod(k, 2)
                idx_v = j * 4 + k
                idx_w = k * 4 + i
                
                for r in range(7):
                    for s in range(7):
                        R = r * 7 + s
                        U444[idx_u, R] += U2[i1 * 2 + j1, r] * U2[i0 * 2 + j0, s]
                        V444[idx_v, R] += V2[j1 * 2 + k1, r] * V2[j0 * 2 + k0, s]
                        W444[idx_w, R] += W2[k1 * 2 + i1, r] * W2[k0 * 2 + i0, s]

    # 3. Construct 4x4x3 decomposition
    # Part A: 4x4x2 (k=0,1 local) -> Global k=4,5
    # Part B: 4x4x1 (k=0 local) -> Global k=6
    
    # 4x4x1 base decomposition
    U441 = np.zeros((16, 16), dtype=float)
    V441 = np.zeros((4, 16), dtype=float)
    W441 = np.zeros((4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U441[i * 4 + j, r] = 1
            V441[j, r] = 1
            W441[i, r] = 1

    # 4x4x2 is two copies of 4x4x1
    U442 = np.zeros((16, 32), dtype=float)
    V442 = np.zeros((8, 32), dtype=float) # m*p = 4*2 = 8
    W442 = np.zeros((8, 32), dtype=float) # p*n = 2*4 = 8
    
    # k=0 -> Global k=4
    for r in range(16):
        U442[:, r] = U441[:, r]
        V442[0::2, r] = V441[:, r] # V index j*2+0
        W442[0::4, r] = W441[:, r] # W index 0*4+i = i
        
    # k=1 -> Global k=5
    for r in range(16):
        U442[:, 16+r] = U441[:, r]
        V442[1::2, 16+r] = V441[:, r] # V index j*2+1
        W442[4::4, 16+r] = W441[:, r] # W index 1*4+i = 4+i

    # 4x4x1 for k=2 local -> Global k=6
    U441_shifted = np.zeros((16, 16), dtype=float)
    V441_shifted = np.zeros((4, 16), dtype=float)
    W441_shifted = np.zeros((4, 16), dtype=float)
    for r in range(16):
        U441_shifted[:, r] = U441[:, r]
        V441_shifted[:, r] = V441[:, r] # V index j*1+0 = j
        W441_shifted[:, r] = W441[:, r] # W index 0*4+i = i

    # Combine 4x4x3 (Rank 48)
    U443 = np.hstack([U442, U441_shifted])
    # V for 4x4x3 has shape (4*3, 48) = (12, 48)
    V443 = np.zeros((12, 48), dtype=float)
    V443[0:8, 0:32] = V442
    V443[8:12, 32:48] = V441_shifted # V index for k=2 is j*3+2? No.
    # Wait, V index is j*p + k.
    # For 4x4x2 part, local p=2. Global k=4,5.
    # V442 rows are 0..7. These correspond to global V indices?
    # No, we are building the full V matrix of size (4*7, Rank).
    # It is easier to build the full U, V, W directly.

    # Let's restart the combination strategy to avoid index confusion.
    # Full U: (16, 97)
    # Full V: (28, 97)
    # Full W: (28, 97)
    
    U_full = np.zeros((16, 97), dtype=float)
    V_full = np.zeros((28, 97), dtype=float)
    W_full = np.zeros((28, 97), dtype=float)
    
    # Part 1: 4x4x4 (Rank 49)
    U_full[:, 0:49] = U444
    V_full[:, 0:49] = V444 # V444 is 16x49. But V_full is 28x97.
    # V444 indices are j*4+k for k in 0..3.
    # These map to global V indices j*7+k.
    # So we need to scatter V444 into V_full.
    for j in range(4):
        for k in range(4):
            idx_v_local = j * 4 + k
            idx_v_global = j * 7 + k
            V_full[idx_v_global, 0:49] = V444[idx_v_local, :]
            
    # W444 indices are k*4+i for k in 0..3.
    # Global W indices are k*4+i for k in 0..6.
    # So for k in 0..3, idx_w_global = k*4+i.
    for k in range(4):
        for i in range(4):
            idx_w_local = k * 4 + i
            idx_w_global = k * 4 + i
            W_full[idx_w_global, 0:49] = W444[idx_w_local, :]

    # Part 2: 4x4x2 (Rank 32) -> Global k=4,5
    # U is same as 4x4x1
    U_full[:, 49:81] = U442
    # V442 indices j*2+k_local for k_local in 0,1.
    # Global k = 4 + k_local.
    # Global V index = j*7 + (4+k_local).
    for j in range(4):
        for k_local in range(2):
            idx_v_local = j * 2 + k_local
            idx_v_global = j * 7 + (4 + k_local)
            V_full[idx_v_global, 49:81] = V442[idx_v_local, :]
            
    # W442 indices k_local*4+i.
    # Global k = 4 + k_local.
    # Global W index = (4+k_local)*4 + i.
    for k_local in range(2):
        for i in range(4):
            idx_w_local = k_local * 4 + i
            idx_w_global = (4 + k_local) * 4 + i
            W_full[idx_w_global, 49:81] = W442[idx_w_local, :]

    # Part 3: 4x4x1 (Rank 16) -> Global k=6
    U_full[:, 81:97] = U441_shifted
    # V441_shifted indices j*1+0 = j.
    # Global k = 6.
    # Global V index = j*7 + 6.
    for j in range(4):
        idx_v_local = j
        idx_v_global = j * 7 + 6
        V_full[idx_v_global, 81:97] = V441_shifted[idx_v_local, :]
        
    # W441_shifted indices 0*4+i = i.
    # Global k = 6.
    # Global W index = 6*4 + i.
    for i in range(4):
        idx_w_local = i
        idx_w_global = 6 * 4 + i
        W_full[idx_w_global, 81:97] = W441_shifted[idx_w_local, :]

    # Convert to lists of lists as required
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
