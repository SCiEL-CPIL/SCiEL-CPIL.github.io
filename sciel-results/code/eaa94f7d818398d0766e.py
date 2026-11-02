import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Hardcoded rank 11 decomposition for 2x3x2
    # U: 6x11, V: 6x11, W: 4x11
    U_232 = np.zeros((6, 11))
    V_232 = np.zeros((6, 11))
    W_232 = np.zeros((4, 11))
    
    # Strassen part (cols 0-6)
    # M1: (A00+A11)(B00+B11)
    U_232[0, 0] = 1; U_232[3, 0] = 1
    V_232[0, 0] = 1; V_232[3, 0] = 1
    W_232[0, 0] = 1; W_232[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U_232[2, 1] = 1; U_232[3, 1] = 1
    V_232[0, 1] = 1
    W_232[1, 1] = 1; W_232[3, 1] = -1 # C10, C11
    
    # M3: A00(B01-B11)
    U_232[0, 2] = 1
    V_232[1, 2] = 1; V_232[3, 2] = -1
    W_232[2, 2] = 1; W_232[3, 2] = 1 # C01, C11
    
    # M4: A11(B10-B00)
    U_232[3, 3] = 1
    V_232[2, 3] = 1; V_232[0, 3] = -1
    W_232[0, 3] = 1; W_232[1, 3] = 1 # C00, C10
    
    # M5: (A00+A01)B11
    U_232[0, 4] = 1; U_232[1, 4] = 1
    V_232[3, 4] = 1
    W_232[0, 4] = -1; W_232[2, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U_232[2, 5] = 1; U_232[0, 5] = -1
    V_232[0, 5] = 1; V_232[1, 5] = 1
    W_232[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U_232[1, 6] = 1; U_232[3, 6] = -1
    V_232[2, 6] = 1; V_232[3, 6] = 1
    W_232[0, 6] = 1 # C00

    # Cross terms (cols 7-10)
    # M8: A02 B20 -> C00
    U_232[2, 7] = 1
    V_232[4, 7] = 1
    W_232[0, 7] = 1
    
    # M9: A02 B21 -> C01
    U_232[2, 8] = 1
    V_232[5, 8] = 1
    W_232[2, 8] = 1
    
    # M10: A12 B20 -> C10
    U_232[5, 9] = 1
    V_232[4, 9] = 1
    W_232[1, 9] = 1
    
    # M11: A12 B21 -> C11
    U_232[5, 10] = 1
    V_232[5, 10] = 1
    W_232[3, 10] = 1

    # Expand to 2x3x6
    repeats = p // 2
    r_sub = 11
    r_total = repeats * r_sub
    
    U_full = np.zeros((n * m, r_total))
    V_full = np.zeros((m * p, r_total))
    W_full = np.zeros((p * n, r_total))
    
    for b in range(repeats):
        offset = b * r_sub
        
        # U is shared
        U_full[:, offset:offset+r_sub] = U_232
        
        # V mapping
        for j in range(m):
            for k in range(2):
                v_idx = j * 2 + k
                v_full_idx = j * p + (b * 2 + k)
                V_full[v_full_idx, offset:offset+r_sub] = V_232[v_idx]
        
        # W mapping
        for i in range(n):
            for k in range(2):
                w_idx = k * 2 + i
                w_full_idx = (b * 2 + k) * n + i
                W_full[w_full_idx, offset:offset+r_sub] = W_232[w_idx]
                
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
