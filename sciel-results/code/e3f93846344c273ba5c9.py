"""Matrix multiplication 3,4,8 rank-88 decomposition with optimized loops."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 block
    # 4 * 8 = 32 terms
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # 2 * 4 * 7 = 56 terms
    
    # Strassen local factors
    strassen_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ]
    strassen_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    strassen_W = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    # Precompute non-zero indices
    U_nz = [[(l, strassen_U[t][l]) for l in range(4) if strassen_U[t][l] != 0] for t in range(7)]
    V_nz = [[(l, strassen_V[t][l]) for l in range(4) if strassen_V[t][l] != 0] for t in range(7)]
    W_nz = [[(l, strassen_W[t][l]) for l in range(4) if strassen_W[t][l] != 0] for t in range(7)]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l, val in U_nz[t]:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    gj = j_start + lj
                    U[gi * m + gj][col] += val
                    
                for l, val in V_nz[t]:
                    lj = l // 2
                    lk = l % 2
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk][col] += val
                    
                for l, val in W_nz[t]:
                    li = l // 2
                    lk = l % 2
                    gi = 1 + li
                    gk = k_start + lk
                    W[gk * n + gi][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
