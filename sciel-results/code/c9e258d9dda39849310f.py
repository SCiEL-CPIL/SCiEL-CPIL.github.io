import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Create local templates for the 11-rank 3x2x2 block
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6
    U_loc[0, 5] = -1; U_loc[2, 5] = 1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Trivial part
    # M8: A4 B0 -> C4
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9: A5 B2 -> C4
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10: A4 B1 -> C5
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11: A5 B3 -> C5
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Stamp the 4 blocks
    # Block 0: a=0, b=0, c=0
    # Block 1: a=0, b=0, c=2
    # Block 2: a=2, b=2, c=0
    # Block 3: a=2, b=2, c=2
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        for r in range(11):
            gr = b_idx * 11 + r
            
            # U
            for i in range(3):
                for j in range(2):
                    val = U_loc[i * 2 + j, r]
                    if val != 0:
                        U[i * m + (a_off + j), gr] = val
            
            # V
            for j in range(2):
                for k in range(2):
                    val = V_loc[j * 2 + k, r]
                    if val != 0:
                        V[(b_off + j) * p + k, gr] = val
            
            # W
            for i in range(3):
                for k in range(2):
                    val = W_loc[i * 2 + k, r]
                    if val != 0:
                        W[(c_off + k) * n + i, gr] = val

    return {"U": U, "V": V, "W": W}
