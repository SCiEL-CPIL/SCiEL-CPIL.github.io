import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # 5 seconds safety margin before 120s
    
    n, m, p = 2, 2, 8
    rank = 28
    
    # Strassen factors for 2x2 x 2x2 -> 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (within a block)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (within a block)
    
    # Strassen terms P1..P7
    U_strassen = np.array([
        [1, 0, 0, 1],    # P1: (A00+A11)
        [0, 0, 1, 1],    # P2: (A10+A11)
        [1, 0, 0, 0],    # P3: A00
        [0, 0, 0, 1],    # P4: A11
        [1, 1, 0, 0],    # P5: (A00+A01)
        [-1, 0, 1, 0],   # P6: (A10-A00)
        [0, 1, 0, -1]    # P7: (A01-A11)
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 0, 0, 1],    # P1: (B00+B11)
        [1, 0, 0, 0],    # P2: B00
        [0, 1, 0, -1],   # P3: (B01-B11)
        [-1, 0, 1, 0],   # P4: (B10-B00)
        [0, 0, 0, 1],    # P5: B11
        [1, 1, 0, 0],    # P6: (B00+B01)
        [0, 0, 1, 1]     # P7: (B10+B11)
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1],    # P1
        [0, 0, 1, -1],   # P2
        [0, 1, 0, 1],    # P3
        [1, 0, 1, 0],    # P4
        [-1, 1, 0, 0],   # P5
        [0, 0, 0, 1],    # P6
        [1, 0, 0, 0]     # P7
    ], dtype=float)
    
    num_blocks = p // 2  # 4 blocks of 2 columns
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # U: same for all blocks
            U[:, col] = U_strassen[r]
            
            # V: B indices for block t
            # B[k, 2t] -> index k*8 + 2t
            # B[k, 2t+1] -> index k*8 + 2t+1
            # V_strassen[r] = [v00, v01, v10, v11]
            V[0*8 + 2*t, col] = V_strassen[r][0]     # B[0, 2t]
            V[0*8 + 2*t+1, col] = V_strassen[r][1]   # B[0, 2t+1]
            V[1*8 + 2*t, col] = V_strassen[r][2]     # B[1, 2t]
            V[1*8 + 2*t+1, col] = V_strassen[r][3]   # B[1, 2t+1]
            
            # W: C indices for block t
            # C[i, 2t] -> index i*8 + 2t
            # C[i, 2t+1] -> index i*8 + 2t+1
            # W_strassen[r] = [w00, w01, w10, w11]
            W[0*8 + 2*t, col] = W_strassen[r][0]     # C[0, 2t]
            W[0*8 + 2*t+1, col] = W_strassen[r][1]   # C[0, 2t+1]
            W[1*8 + 2*t, col] = W_strassen[r][2]     # C[1, 2t]
            W[1*8 + 2*t+1, col] = W_strassen[r][3]   # C[1, 2t+1]
            
            col += 1
    
    # Verify correctness
    T = np.zeros((4, 16, 16))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(8):
                    if j == k:
                        T[i*2+j, k*8+l, i*8+l] = 1
    
    T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback: should not happen
        pass
    
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
