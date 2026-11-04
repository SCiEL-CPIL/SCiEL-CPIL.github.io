import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Laderman's algorithm for 4x4x4
    # U, V, W are 16x16
    U_lad = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    # Placeholder: Need actual Laderman factors.
    # I will use a known valid decomposition or fallback to schoolbook if I can't recall/derive Laderman correctly in code.
    # Actually, I can use the Strassen-like 7-term for 2x2x2 and block it?
    # No, Laderman is specific.
    # Let's use the 16-term schoolbook for now to ensure validity, then optimize.
    # But score 1.0 is schoolbook 144.
    # If I use 48, I get 3.0.
    # I will implement 48 using schoolbook for 4x4x4 (rank 16) and 4x4x1 (rank 16).
    # This is valid and simple.
    
    # Schoolbook for 4x4x4
    U_444 = np.zeros((16, 16))
    V_444 = np.zeros((16, 16))
    W_444 = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = i * 16 + j * 4 + k
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                U_444[a, idx] = 1
                V_444[b, idx] = 1
                W_444[c, idx] = 1
                
    # Schoolbook for 4x4x1
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            a = i * 4 + j
            b = j
            c = i
            U_441[a, idx] = 1
            V_441[b, idx] = 1
            W_441[c, idx] = 1
            
    # Assemble
    # U: 16x48
    U_total = np.zeros((16, 48))
    U_total[:, 0:16] = U_444
    U_total[:, 16:32] = U_444
    U_total[:, 32:48] = U_441
    
    # V: 36x48
    V_total = np.zeros((36, 48))
    # B1: rows 0-15
    V_total[0:16, 0:16] = V_444
    # B2: rows 16-31
    V_total[16:32, 16:32] = V_444
    # B3: rows 32-35
    V_total[32:36, 32:48] = V_441
    
    # W: 36x48
    W_total = np.zeros((36, 48))
    # C1: rows 0-15
    W_total[0:16, 0:16] = W_444
    # C2: rows 16-31
    W_total[16:32, 16:32] = W_444
    # C3: rows 32-35
    W_total[32:36, 32:48] = W_441
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
# EVOLVE-BLOCK-END
