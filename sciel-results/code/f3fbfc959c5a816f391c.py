import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen decomposition for 2x2
    # U, V, W are 4x7 matrices
    # Indices: U[i, j], V[j, k], W[k, i]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Rank 0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # Rank 1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[1, 0] = 1                # C10 (k=0, i=1)
    
    # Rank 2: M3 = A00(B01-B11)
    U2[0, 2] = 1                # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, B11
    W2[2, 0] = 1                # C01 (k=1, i=0)
    
    # Rank 3: M4 = A11(B10-B00)
    U2[3, 3] = 1                # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, B00
    W2[0, 0] = 1                # C00 (k=0, i=0)
    
    # Rank 4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1                # B11
    W2[0, 0] = -1; W2[2, 0] = 1 # C00, C01
    
    # Rank 5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1  # B00, B01
    W2[3, 0] = 1                # C11 (k=1, i=1)
    
    # Rank 6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 0] = 1                # C00 (k=0, i=0)
    
    # Construct 4x4 decomposition via Kronecker product
    # Rank is 7*7 = 49
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            I = 2 * i1 + i2
                            J = 2 * j1 + j2
                            idx = I * 4 + J
                            U[idx, c] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    K = 2 * k1 + k2
                                    idx_V = J * 4 + K
                                    V[idx_V, c] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    
                                    idx_W = K * 4 + I
                                    W[idx_W, c] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
                                    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
