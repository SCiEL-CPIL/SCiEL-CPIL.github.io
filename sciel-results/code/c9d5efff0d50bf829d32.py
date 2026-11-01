import time

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [1, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1]
    ]
    
    V_s = [
        [1, 0, 1, -1, 0, 1, 1],
        [1, 1, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    for block in range(4):
        offset = block * 7
        for i in range(4):
            for k in range(7):
                U[i][offset + k] = U_s[i][k]
                for r in range(2):
                    for c in range(2):
                        local_idx = r * 2 + c
                        global_idx = r * 8 + (block * 2 + c)
                        V[global_idx][offset + k] = V_s[local_idx][k]
                        w_idx = (block * 2 + c) * 2 + r
                        W[w_idx][offset + k] = W_s[local_idx][k]
                        
    return {"U": U, "V": V, "W": W}
