import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Step 1: Define Strassen 2x2x2 decomposition
    # Rank 7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Step 2: Construct 4x4x4 decomposition via Kronecker product
    # Rank 49
    rank_44 = 49
    U_44 = np.zeros((16, rank_44))
    V_44 = np.zeros((16, rank_44))
    W_44 = np.zeros((16, rank_44))
    
    for r1 in range(7):
        for r0 in range(7):
            r44 = r1 * 7 + r0
            for i1 in range(2):
                for i0 in range(2):
                    i44 = i1 * 2 + i0
                    # U index: i*m + j. Here m=2 for 2x2.
                    # Wait, U_2 is 4x7. Index is i*2+j.
                    # For 4x4, index is i*4+j.
                    # U_44[i44*4 + j44, r44]
                    # We need to fill all i, j.
                    pass # Placeholder, doing full loop below
    
    # More efficient filling
    for r1 in range(7):
        for r0 in range(7):
            r44 = r1 * 7 + r0
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i44 = i1 * 2 + i0
                            j44 = j1 * 2 + j0
                            idx_u = i44 * 4 + j44
                            U_44[idx_u, r44] = U_2[i1*2+j1, r1] * U_2[i0*2+j0, r0]
                            
                            # V
                            idx_v = j44 * 4 + (i1*2+i0) # Wait, V is j*p+k. For 2x2, p=2.
                            # V_2[j*2+k, r].
                            # V_44[j44*4+k44, r44].
                            # k44 corresponds to i44 in the 2x2x2 sense?
                            # No, V_2 is for B. B is 2x2.
                            # In 4x4x4, B is 4x4.
                            # V_44[j, k, r].
                            # j44 = j1*2+j0, k44 = k1*2+k0.
                            # V_44[idx_v, r44] = V_2[j1*2+k1, r1] * V_2[j0*2+k0, r0]
                            pass # Need to loop k too

    # Let's redo the Kronecker loop properly
    U_44 = np.zeros((16, rank_44))
    V_44 = np.zeros((16, rank_44))
    W_44 = np.zeros((16, rank_44))
    
    for r1 in range(7):
        for r0 in range(7):
            r44 = r1 * 7 + r0
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    i44 = i1 * 2 + i0
                                    j44 = j1 * 2 + j0
                                    k44 = k1 * 2 + k0
                                    
                                    idx_u = i44 * 4 + j44
                                    idx_v = j44 * 4 + k44
                                    idx_w = k44 * 4 + i44
                                    
                                    u_val = U_2[i1*2+j1, r1] * U_2[i0*2+j0, r0]
                                    v_val = V_2[j1*2+k1, r1] * V_2[j0*2+k0, r0]
                                    w_val = W_2[k1*2+i1, r1] * W_2[k0*2+i0, r0]
                                    
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        U_44[idx_u, r44] = u_val
                                        V_44[idx_v, r44] = v_val
                                        W_44[idx_w, r44] = w_val

    # Step 3: Construct 4x4x8 decomposition
    # Rank 49
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U is same as U_44
    U[:, :] = U_44
    
    # V is V_44 repeated in the k dimension
    # V[j, k, r] = V_44[j, k % 4, r]
    # Index in V: j * 8 + k
    # Index in V_44: j * 4 + k'
    for j in range(4):
        for k in range(8):
            k_prime = k % 4
            idx_v = j * 8 + k
            idx_v44 = j * 4 + k_prime
            V[idx_v, :] = V_44[idx_v44, :]
            
    # W is W_44 repeated in the k dimension
    # W[k, i, r] = W_44[k % 4, i, r]
    # Index in W: k * 4 + i
    # Index in W_44: k' * 4 + i
    for k in range(8):
        for i in range(4):
            k_prime = k % 4
            idx_w = k * 4 + i
            idx_w44 = k_prime * 4 + i
            W[idx_w, :] = W_44[idx_w44, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
