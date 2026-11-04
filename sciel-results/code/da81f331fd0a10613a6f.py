import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # ... (same initialization as before) ...
    
    # ... (same specs) ...
    
    for half in range(2):
        if time.time() > deadline:
            break
        for k_p in range(7):
            if time.time() > deadline:
                break
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                # ... (same filling logic) ...
                # Check time less frequently
                if (col % 10 == 0) and time.time() > deadline:
                    break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
