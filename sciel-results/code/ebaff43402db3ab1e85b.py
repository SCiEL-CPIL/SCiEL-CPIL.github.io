import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # r=0..6
    # U[a,b,r], V[b,c,r], W[c,a,r]
    # a,b,c in {0,1}
    
    U_local = np.zeros((2, 2, 7), dtype=float)
    V_local = np.zeros((2, 2, 7), dtype=float)
    W_local = np.zeros((2, 2, 7), dtype=float)
    
    # r=0 (M1)
    U_local[0,0,0] = 1; U_local[1,1,0] = 1
    V_local[0,0,0] = 1; V_local[1,1,0] = 1
    W_local[0,0,0] = 1; W_local[1,1,0] = 1
    
    # r=1 (M2)
    U_local[1,0,1] = 1; U_local[1,1,1] = 1
    V_local[0,0,1] = 1
    W_local[0,1,1] = 1; W_local[1,1,1] = -1
    
    # r=2 (M3)
    U_local[0,0,2] = 1
    V_local[0,1,2] = 1; V_local[1,1,2] = -1
    W_local[1,0,2] = 1; W_local[1,1,2] = 1
    
    # r=3 (M4)
    U_local[1,1,3] = 1
    V_local[1,0,3] = 1; V_local[0,0,3] = -1
    W_local[0,0,3] = 1; W_local[0,1,3] = 1
    
    # r=4 (M5)
    U_local[0,0,4] = 1; U_local[0,1,4] = 1
    V_local[1,1,4] = 1
    W_local[0,0,4] = -1; W_local[1,0,4] = 1
    
    # r=5 (M6)
    U_local[1,0,5] = 1; U_local[0,0,5] = -1
    V_local[0,0,5] = 1; V_local[0,1,5] = 1
    W_local[1,1,5] = 1
    
    # r=6 (M7)
    U_local[0,1,6] = 1; U_local[1,1,6] = -1
    V_local[1,0,6] = 1; V_local[1,1,6] = 1
    W_local[0,0,6] = 1
    
    # Blocks
    # j_A0, j_B0, k0
    blocks = [
        (0, 0, 0), # j in {0,1}, k in {0,1}
        (0, 0, 2), # j in {0,1}, k in {2,3}
        (0, 0, 4), # j in {0,1}, k in {4,5}
        (2, 2, 0), # j in {2,3}, k in {0,1}
        (2, 2, 2), # j in {2,3}, k in {2,3}
        (2, 2, 4), # j in {2,3}, k in {4,5}
    ]
    
    for b_idx, (jA0, jB0, k0) in enumerate(blocks):
        for r in range(7):
            col = b_idx * 7 + r
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        # Global indices
                        i = a
                        jA = jA0 + b
                        jB = jB0 + b
                        k = k0 + c
                        
                        I = i * m + jA
                        J = jB * p + k
                        K = k * n + i
                        
                        U[I, col] += U_local[a, b, r]
                        V[J, col] += V_local[b, c, r]
                        W[K, col] += W_local[c, a, r]
                        
    return {"U": U, "V": V, "W": W}
