import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block sizes
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 3, 3, 3
    
    # 1. Construct M_{2,2,2} decomposition (Strassen, rank 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    V2 = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    W2 = [
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, -1], # P2 -> C10, C11(-)
        [0, 1, 0, 1], # P3 -> C01, C11
        [1, 0, 1, 0], # P4 -> C00, C10
        [-1, 1, 0, 0], # P5 -> C00(-), C01
        [0, 0, 0, 1], # P6 -> C11
        [1, 0, 0, 0]  # P7 -> C00
    ]
    
    # 2. Construct M_{3,3,3} decomposition (Schoolbook, rank 27)
    # Indices: i*3 + j
    U3 = []
    V3 = []
    W3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3 + j] = 1
                v[j*3 + k] = 1
                w[k*3 + i] = 1
                U3.append(u)
                V3.append(v)
                W3.append(w)
                
    rank = 7 * 27
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    col = 0
    for a in range(7):
        for b in range(27):
            # Fill U
            for I in range(2):
                for J in range(2):
                    val2 = U2[a][I*2 + J]
                    if val2 == 0: continue
                    for i in range(3):
                        for j in range(3):
                            val3 = U3[b][i*3 + j]
                            if val3 == 0: continue
                            val = val2 * val3
                            idx = I*18 + i*6 + J*3 + j
                            U[idx][col] = val
                            
            # Fill V
            for J in range(2):
                for K in range(2):
                    val2 = V2[a][J*2 + K]
                    if val2 == 0: continue
                    for j in range(3):
                        for k in range(3):
                            val3 = V3[b][j*3 + k]
                            if val3 == 0: continue
                            val = val2 * val3
                            idx = J*18 + j*6 + K*3 + k
                            V[idx][col] = val
                            
            # Fill W
            for K in range(2):
                for I in range(2):
                    val2 = W2[a][K*2 + I]
                    if val2 == 0: continue
                    for k in range(3):
                        for i in range(3):
                            val3 = W3[b][k*3 + i]
                            if val3 == 0: continue
                            val = val2 * val3
                            idx = K*18 + k*6 + I*3 + i
                            W[idx][col] = val
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
