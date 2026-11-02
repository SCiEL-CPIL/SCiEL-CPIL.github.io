import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    blocks = [
        {'A_map': [0, 1, 4, 5], 'B_map': [0, 1, 4, 5], 'C_map': [0, 1, 4, 5]},
        {'A_map': [2, 3, 6, 7], 'B_map': [8, 9, 12, 13], 'C_map': [0, 1, 4, 5]},
        {'A_map': [0, 1, 4, 5], 'B_map': [2, 3, 6, 7], 'C_map': [2, 3, 6, 7]},
        {'A_map': [2, 3, 6, 7], 'B_map': [10, 11, 14, 15], 'C_map': [2, 3, 6, 7]}
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        A_map = block['A_map']
        B_map = block['B_map']
        C_map = block['C_map']
        
        for u_loc, v_loc, w_loc in strassen_terms:
            if time.time() > deadline:
                break
            for i in range(4):
                if u_loc[i] != 0:
                    U[A_map[i]][col] += u_loc[i]
            for i in range(4):
                if v_loc[i] != 0:
                    V[B_map[i]][col] += v_loc[i]
            for i in range(4):
                if w_loc[i] != 0:
                    W[C_map[i]][col] += w_loc[i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
