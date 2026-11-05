import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 3, 4, 5
    U = np.zeros((n * m, 0), dtype=float)
    V = np.zeros((m * p, 0), dtype=float)
    W = np.zeros((p * n, 0), dtype=float)
    
    def add_block(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1):
        nonlocal U, V, W
        if time.time() > deadline:
            return
        ar = a_r1 - a_r0
        ac = a_c1 - a_c0
        bc = b_c1 - b_c0
        rank = ar * ac * bc
        U_new = np.zeros((n * m, rank))
        V_new = np.zeros((m * p, rank))
        W_new = np.zeros((p * n, rank))
        col = 0
        for i in range(a_r0, a_r1):
            for j in range(a_c0, a_c1):
                for k in range(b_c0, b_c1):
                    U_new[i * m + j, col] = 1.0
                    V_new[j * p + k, col] = 1.0
                    W_new[k * n + i, col] = 1.0
                    col += 1
        U = np.hstack([U, U_new])
        V = np.hstack([V, V_new])
        W = np.hstack([W, W_new])
    
    # Partition: A rows [0,2), [2,3), A cols [0,2), [2,4)
    # B rows [0,2), [2,4), B cols [0,2), [2,4), [4,5)
    blocks = [
        (0, 2, 0, 2, 0, 2, 0, 2),
        (0, 2, 0, 2, 0, 2, 2, 4),
        (0, 2, 0, 2, 0, 2, 4, 5),
        (0, 2, 2, 4, 2, 4, 0, 2),
        (0, 2, 2, 4, 2, 4, 2, 4),
        (0, 2, 2, 4, 2, 4, 4, 5),
        (2, 3, 0, 2, 0, 2, 0, 2),
        (2, 3, 0, 2, 0, 2, 2, 4),
        (2, 3, 0, 2, 0, 2, 4, 5),
        (2, 3, 2, 4, 2, 4, 0, 2),
        (2, 3, 2, 4, 2, 4, 2, 4),
        (2, 3, 2, 4, 2, 4, 4, 5),
    ]
    
    for block in blocks:
        add_block(*block)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
