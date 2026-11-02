import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(2,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),
        ([(0,1)], [(2,1),(3,-1)], [(3,-1)]),
        ([(0,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(1,1)]),
        ([(1,1),(3,1)], [(3,1)], [(1,1),(3,1)]),
        ([(0,1),(1,1)], [(0,1),(2,-1)], [(0,1)]),
        ([(2,1),(0,-1)], [(1,1),(3,1)], [(0,1),(3,1)])
    ]
    
    for pair_idx in range(4):
        if time.time() > deadline:
            break
        j = pair_idx * 2
        b00 = j
        b01 = j + 1
        b10 = 9 + j
        b11 = 9 + j + 1
        c00 = j
        c01 = j + 1
        c10 = 9 + j
        c11 = 9 + j + 1
        
        for term in strassen_terms:
            u_list, v_list, w_list = term
            for u_idx, u_val in u_list:
                U[u_idx][col] += u_val
            for v_idx, v_val in v_list:
                V[v_idx][col] += v_val
            for w_idx, w_val in w_list:
                W[w_idx][col] += w_val
            col += 1
            
    if time.time() <= deadline:
        # Schoolbook for last column (j=8)
        U[0][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        U[1][col] = 1; V[17][col] = 1; W[8][col] = 1; col += 1
        U[2][col] = 1; V[8][col] = 1; W[17][col] = 1; col += 1
        U[3][col] = 1; V[17][col] = 1; W[17][col] = 1; col += 1
        
        # Outer product for last row
        for j in range(9):
            if time.time() > deadline:
                break
            U[4][col] = 1
            V[18+j][col] = 1
            W[j][col] = 1
            col += 1
            U[5][col] = 1
            V[18+j][col] = 1
            W[9+j][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
