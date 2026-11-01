import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s wall clock limit, use 148s for safety
    
    n, m, p = DIMENSIONS
    R = 28
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, -1]
    ]
    
    num_blocks = 4
    
    for r in range(num_blocks):
        for k in range(7):
            col = r * 7 + k
            
            for i in range(4):
                U[i][col] = u[k][i]
                
            for l in range(4):
                j = l // 2
                k_loc = l % 2
                v_row = j * p + (2 * r + k_loc)
                V[v_row][col] = v[k][l]
                
            for l in range(4):
                i = l // 2
                k_loc = l % 2
                w_row = (2 * r + k_loc) * n + i
                W[w_row][col] = w[k][l]
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
