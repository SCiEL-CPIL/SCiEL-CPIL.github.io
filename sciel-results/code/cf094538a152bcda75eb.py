import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 Strassen decomposition for 2x2 matrices
    # U_S, V_S, W_S are 4x7 matrices
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # r=0: P1 = (A00+A11)(B00+B11)
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # r=1: P2 = (A10+A11)B00
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = -1
    
    # r=2: P3 = A00(B01-B11)
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1; W_S[3, 2] = 1
    
    # r=3: P4 = A11(B10-B00)
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = 1; W_S[2, 3] = 1
    
    # r=4: P5 = (A00+A01)B11
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[1, 4] = 1
    
    # r=5: P6 = (A10-A00)(B00+B01)
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[3, 5] = 1
    
    # r=6: P7 = (A01-A11)(B10+B11)
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1

    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct U
    for I in range(2):
        for J in range(2):
            for i_in in range(2):
                for j_in in range(2):
                    row = (I * 2 + i_in) * 4 + (J * 2 + j_in)
                    idx_block = I * 2 + J
                    idx_inner = i_in * 2 + j_in
                    for r_block in range(7):
                        val_block = U_S[idx_block, r_block]
                        if val_block == 0: continue
                        for r_inner in range(7):
                            val_inner = U_S[idx_inner, r_inner]
                            if val_inner == 0: continue
                            col = r_block * 7 + r_inner
                            U[row, col] = val_block * val_inner
                            
    # Construct V
    for J in range(2):
        for K in range(2):
            for j_in in range(2):
                for k_in in range(2):
                    row = (J * 2 + j_in) * 4 + (K * 2 + k_in)
                    idx_block = J * 2 + K
                    idx_inner = j_in * 2 + k_in
                    for r_block in range(7):
                        val_block = V_S[idx_block, r_block]
                        if val_block == 0: continue
                        for r_inner in range(7):
                            val_inner = V_S[idx_inner, r_inner]
                            if val_inner == 0: continue
                            col = r_block * 7 + r_inner
                            V[row, col] = val_block * val_inner

    # Construct W
    for I in range(2):
        for K in range(2):
            for i_in in range(2):
                for k_in in range(2):
                    # W index is k*n + i. Here n=4.
                    # k = K * 2 + k_in
                    # i = I * 2 + i_in
                    row = (K * 2 + k_in) * 4 + (I * 2 + i_in)
                    idx_block = I * 2 + K
                    idx_inner = i_in * 2 + k_in
                    for r_block in range(7):
                        val_block = W_S[idx_block, r_block]
                        if val_block == 0: continue
                        for r_inner in range(7):
                            val_inner = W_S[idx_inner, r_inner]
                            if val_inner == 0: continue
                            col = r_block * 7 + r_inner
                            W[row, col] = val_block * val_inner

    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
