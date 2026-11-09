import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    # We'll build the decomposition using Strassen for 2x2x2 sub-blocks
    # and schoolbook for the 2x4x1 part.
    
    # Strassen 2x2x2 coefficients (U, V, W) for local indices
    # Local: i,j,k in {0,1}
    # U: [i*2+j] -> 4 entries
    # V: [j*2+k] -> 4 entries
    # W: [k*2+i] -> 4 entries
    
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],  # M1: C00+=1, C11+=1
        [0,1,0,-1], # M2: C10+=1, C11+=-1
        [0,0,1,1],  # M3: C01+=1, C11+=1
        [1,1,0,0],  # M4: C00+=1, C10+=1
        [-1,0,1,0], # M5: C00+=-1, C01+=1
        [0,0,0,1],  # M6: C11+=1
        [1,0,0,0],  # M7: C00+=1
    ]
    
    # Total rank: 4 sub-problems * 7 + 8 (schoolbook for k=4) = 36
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 20 x rank
    W = [[0] * rank for _ in range(p * n)]  # 10 x rank
    
    col = 0
    # Four 2x2x2 Strassen sub-problems
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
