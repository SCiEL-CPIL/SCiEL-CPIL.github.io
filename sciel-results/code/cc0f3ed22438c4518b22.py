import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)] # 6 rows
    V = [[0] * rank for _ in range(m * p)] # 9 rows
    W = [[0] * rank for _ in range(p * n)] # 6 rows
    
    # T1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # T2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # T3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # T4
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # T5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # T6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # T7
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # T8
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # T9
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # T10
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # T11
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    # T12
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1
    
    # T13
    U[2][12] = 1
    V[7][12] = 1
    W[2][12] = 1
    
    # T14
    U[2][13] = 1
    V[8][13] = 1
    W[4][13] = 1
    
    # T15
    U[5][14] = 1
    V[6][14] = 1
    W[1][14] = 1
    
    # T16
    U[5][15] = 1
    V[7][15] = 1
    W[3][15] = 1
    
    # T17
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    if time.time() > deadline:
        # Should never happen, but satisfy directive
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
