import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # 2x2x2 Strassen decomposition
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11) -> C00+C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # c=1: (A10+A11)B00 -> C10-C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # c=2: A00(B01-B11) -> C01+C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # c=3: A11(B10-B00) -> C00+C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # c=4: (A00+A01)B11 -> -C00+C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # c=5: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # c=6: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)]
    ]
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)]
    ]
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)]
    ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for kp in range(7):
        for ms in range(7):
            col = kp * 7 + ms
            for coeff, block_idx in A_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gi = 2 * bi + r
                        gj = 2 * bj + c
                        U4[gi * 4 + gj, col] += coeff * U2[r * 2 + c, ms]
            for coeff, block_idx in B_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gj = 2 * bi + r
                        gk = 2 * bj + c
                        V4[gj * 4 + gk, col] += coeff * V2[r * 2 + c, ms]
            for coeff, block_idx in C_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gi = 2 * bi + r
                        gk = 2 * bj + c
                        W4[gk * 4 + gi, col] += coeff * W2[r * 2 + c, ms]
    
    # Extend to 4x4x8: two independent 4x4x4 multiplications
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[:, :49] = U4
    U[:, 49:] = U4
    
    for j in range(4):
        for k_local in range(4):
            V[j * 8 + k_local, :49] = V4[j * 4 + k_local, :]
            V[j * 8 + (4 + k_local), 49:] = V4[j * 4 + k_local, :]
    
    for k_local in range(4):
        for i in range(4):
            W[k_local * 4 + i, :49] = W4[k_local * 4 + i, :]
            W[(4 + k_local) * 4 + i, 49:] = W4[k_local * 4 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
