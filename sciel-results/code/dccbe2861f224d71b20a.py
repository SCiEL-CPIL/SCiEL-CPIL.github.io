import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x1 Trivial (Rank 16)
    # Part 3: 4x1x4 Trivial (Rank 16)
    # Part 4: 4x1x1 Trivial (Rank 4)
    total_rank = 49 + 16 + 16 + 4
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # --- Part 1: 4x4x4 Strassen ---
    # Get 2x2 Strassen coefficients
    U2, V2, W2 = get_strassen_2x2()
    
    # Build 4x4 coefficients via Kronecker
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    u_idx = i*4 + j
                    # Local indices for 2x2
                    u_idx_1 = i1*2 + j1
                    u_idx_0 = i0*2 + j0
                    
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            val = U2[u_idx_1, r1] * U2[u_idx_0, r2]
                            U4[u_idx, r] = val
                            
                            # V: j, k
                            # j = 2*j1 + j0, k = 2*k1 + k0
                            # We need to iterate k too?
                            # No, V4[j, k, r] depends on j1,k1,r1 and j0,k0,r2
                            # But here we are filling U4.
                            # We need to fill V4 and W4 similarly.
                            
    # This nested loop is inefficient and I missed filling V4 and W4 correctly in the loop above.
    # Let's rewrite the Kronecker construction properly.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            u_idx = i*4 + j
                            u_val = U2[i1*2+j1, r1] * U2[i0*2+j0, r2]
                            U4[u_idx, r] = u_val
                            
                            # V indices
                            # j is fixed from above? No, V depends on j, k.
                            # The Kronecker product applies to all dimensions.
                            # V4[j, k, r] = V2[j1, k1, r1] * V2[j0, k0, r2]
                            # We need to iterate k1, k0 as well.
                            
    # Correct approach:
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i0*2+j0, r2]
                            
                            for k1 in range(2):
                                for k0 in range(2):
                                    k = 2*k1 + k0
                                    V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j0*2+k0, r2]
                                    W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k0*2+i0, r2]
                                    
    # Map U4, V4, W4 to global U, V, W for the 4x4x4 part
    # Global indices:
    # U: i (0..3), j (0..3) -> i*5 + j
    # V: j (0..3), k (0..3) -> j*5 + k
    # W: k (0..3), i (0..3) -> k*4 + i
    
    for idx in range(16):
        # U4 idx is i*4+j
        i = idx // 4
        j = idx % 4
        U[i*5+j, 0:49] += U4[idx, :]
        
        # V4 idx is j*4+k
        j = idx // 4
        k = idx % 4
        V[j*5+k, 0:49] += V4[idx, :]
        
        # W4 idx is k*4+i
        k = idx // 4
        i = idx % 4
        W[k*4+i, 0:49] += W4[idx, :]

    # --- Part 2: 4x4x1 (A_4x4 * b_4x1) ---
    # i in 0..3, j in 0..3, k=4
    # Rank 16
    offset = 49
    for i in range(4):
        for j in range(4):
            r = offset + i*4 + j
            U[i*5+j, r] = 1
            V[j*5+4, r] = 1
            W[4*4+i, r] = 1

    # --- Part 3: 4x1x4 (a_4x1 * c_1x4) ---
    # i in 0..3, j=4, k in 0..3
    # Rank 16
    offset = 49 + 16
    for i in range(4):
        for k in range(4):
            r = offset + i*4 + k
            U[i*5+4, r] = 1
            V[4*5+k, r] = 1
            W[k*4+i, r] = 1

    # --- Part 4: 4x1x1 (a_4x1 * d_1x1) ---
    # i in 0..3, j=4, k=4
    # Rank 4
    offset = 49 + 16 + 16
    for i in range(4):
        r = offset + i
        U[i*5+4, r] = 1
        V[4*5+4, r] = 1
        W[4*4+i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_strassen_2x2():
    # ... (as defined above) ...
    pass
