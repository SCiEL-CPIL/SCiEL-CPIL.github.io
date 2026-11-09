import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Base Strassen 2x2 decomposition
    # Indices:
    # U_base: row = u*2+v, col = m (0-6)
    # V_base: row = v*2+w, col = m (0-6)
    # W_base: row = w*2+u, col = m (0-6)
    
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))

    # m=0 (M1)
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    W_base[0, 0] = 1; W_base[3, 0] = 1

    # m=1 (M2)
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    W_base[2, 1] = 1; W_base[3, 1] = -1

    # m=2 (M3)
    U_base[0, 2] = 1
    V_base[2, 2] = 1; V_base[3, 2] = -1
    W_base[2, 2] = 1; W_base[3, 2] = 1

    # m=3 (M4)
    U_base[3, 3] = 1
    V_base[1, 3] = 1; V_base[0, 3] = -1
    W_base[0, 3] = 1; W_base[1, 3] = 1

    # m=4 (M5)
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[3, 4] = 1
    W_base[0, 4] = -1; W_base[2, 4] = 1

    # m=5 (M6)
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[2, 5] = 1
    W_base[3, 5] = 1

    # m=6 (M7)
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[1, 6] = 1; V_base[3, 6] = 1
    W_base[0, 6] = 1

    # High-level Strassen coefficients
    # c_A[s][ub][vb], c_B[s][vb][wb], c_C[s][ub][wb]
    # s is 0-6
    
    c_A = [
        [[1, 0], [0, 1]], # s=0: A00+A11
        [[0, 0], [1, 1]], # s=1: A10+A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00+A01
        [[-1, 0], [1, 0]],# s=5: A10-A00
        [[0, 1], [0, -1]] # s=6: A01-A11
    ]
    
    c_B = [
        [[1, 0], [0, 1]], # s=0: B00+B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 0], [1, -1]],# s=2: B10-B11
        [[-1, 1], [0, 0]],# s=3: B01-B00
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 0], [1, 0]], # s=5: B00+B10
        [[0, 1], [0, 1]]  # s=6: B01+B11
    ]
    
    c_C = [
        [[1, 0], [0, 1]], # s=0: C00+C11
        [[0, 0], [1, -1]],# s=1: C10-C11
        [[0, 1], [0, 1]], # s=2: C01+C11
        [[1, 0], [1, 0]], # s=3: C00+C10
        [[-1, 1], [0, 0]],# s=4: -C00+C01
        [[0, 0], [0, 1]], # s=5: C11
        [[1, 0], [0, 0]]  # s=6: C00
    ]

    # Fill U, V, W
    for s in range(7):
        for mb in range(7):
            r = s * 7 + mb
            
            # U
            for I in range(4):
                ub = I // 2
                u = I % 2
                for J in range(4):
                    vb = J // 2
                    v = J % 2
                    val = c_A[s][ub][vb] * U_base[u * 2 + v, mb]
                    if val != 0:
                        U[I * 4 + J][r] = val
                        
            # V
            for J in range(4):
                vb = J // 2
                v = J % 2
                for K in range(4):
                    wb = K // 2
                    w = K % 2
                    val = c_B[s][vb][wb] * V_base[v * 2 + w, mb]
                    if val != 0:
                        V[J * 4 + K][r] = val
                        
            # W
            for K in range(4):
                wb = K // 2
                w = K % 2
                for I in range(4):
                    ub = I // 2
                    u = I % 2
                    val = c_C[s][ub][wb] * W_base[w * 2 + u, mb]
                    if val != 0:
                        W[K * 4 + I][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
