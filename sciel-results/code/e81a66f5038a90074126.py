import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin
    
    n, m, p = 6, 6, 6
    
    # Strassen 2x2x2 rank 7 factors
    # U2 shape (4,7): row i*2+j
    # V2 shape (4,7): row j*2+k  
    # W2 shape (4,7): row k*2+i
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # i=0,j=0
        [0, 0, 0, 1, 1, 0, 1],  # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0
        [1, 1, 0, 1, 0, 0, 1],  # i=1,j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 1],  # j=0,k=0
        [0, 0, 1, 1, 0, 0, 1],  # j=0,k=1
        [1, 0, 1, 0, 1, 0, 0],  # j=1,k=0
        [0, 1, 0, 0, 1, 1, 1],  # j=1,k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # k=0,i=0
        [0, 0, 1, 1, 0, 0, 0],  # k=0,i=1
        [0, 1, 0, 0, 0, 1, 0],  # k=1,i=0
        [1, 0, 0, 0, 1, 0, 1],  # k=1,i=1
    ], dtype=float)
    
    # Verify Strassen
    T2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1
    
    T2_recon = np.einsum('ar,br,cr->abc', U2, V2, W2)
    # Check
    if not np.allclose(T2_recon, T2):
        # Try to fix - use a known correct decomposition
        pass
    
    # CW23 3x3x3 rank 23 factors
    # ... (need to define)
    
    # Construct 6x6x6
    rank = 7 * 23
    U6 = np.zeros((36, rank))
    V6 = np.zeros((36, rank))
    W6 = np.zeros((36, rank))
    
    for i1 in range(3):
        for j1 in range(3):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    a = i*6 + j
                    for r1 in range(23):
                        for r2 in range(7):
                            r = r1*7 + r2
                            U6[a, r] = U3[i1*3+j1, r1] * U2[i2*2+j2, r2]
    
    # Similarly for V6, W6
    # ...
    
    # Convert to lists
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
