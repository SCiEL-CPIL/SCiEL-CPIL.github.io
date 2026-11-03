import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 2, 4, 8

    # Build the standard Strassen 2x2x2 decomposition (rank 7)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # r=0 (M1) = (A00+A11)(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1

    # r=1 (M2) = (A10+A11)B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1

    # r=2 (M3) = A00(B10-B11)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1

    # r=3 (M4) = A11(B01-B00)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1

    # r=4 (M5) = (A00+A01)B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1

    # r=5 (M6) = (A10-A00)(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # r=6 (M7) = (A01-A11)(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    # Build the 28-rank decomposition by splitting B into two 4x4 blocks
    total_rank = 28
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)

    # Define the 4 blocks
    # Block 0: A_L (0,1,4,5) * B_TL (0,1,8,9) -> C[:, 0:2] (0,1,2,3)
    u_idx_0 = [0, 1, 4, 5]
    v_idx_0 = [0, 1, 8, 9]
    w_idx_0 = [0, 1, 2, 3]
    
    # Block 1: A_R (2,3,6,7) * B_BL (16,17,24,25) -> C[:, 0:2] (0,1,2,3)
    u_idx_1 = [2, 3, 6, 7]
    v_idx_1 = [16, 17, 24, 25]
    w_idx_1 = [0, 1, 2, 3]
    
    # Block 2: A_L (0,1,4,5) * B_TR (2,3,10,11) -> C[:, 2:4] (4,5,6,7)
    u_idx_2 = [0, 1, 4, 5]
    v_idx_2 = [2, 3, 10, 11]
    w_idx_2 = [4, 5, 6, 7]
    
    # Block 3: A_R (2,3,6,7) * B_BR (18,19,26,27) -> C[:, 2:4] (4,5,6,7)
    u_idx_3 = [2, 3, 6, 7]
    v_idx_3 = [18, 19, 26, 27]
    w_idx_3 = [4, 5, 6, 7]

    blocks = [
        (u_idx_0, v_idx_0, w_idx_0),
        (u_idx_1, v_idx_1, w_idx_1),
        (u_idx_2, v_idx_2, w_idx_2),
        (u_idx_3, v_idx_3, w_idx_3)
    ]

    for b in range(4):
        u_map, v_map, w_map = blocks[b]
        base_r = b * 7
        for l in range(7):
            r = base_r + l
            for loc in range(4):
                val_u = U_local[loc, l]
                if val_u != 0:
                    U[u_map[loc], r] += val_u
                
                val_v = V_local[loc, l]
                if val_v != 0:
                    V[v_map[loc], r] += val_v
                
                val_w = W_local[loc, l]
                if val_w != 0:
                    W[w_map[loc], r] += val_w

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to naive 64 if something went wrong
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
