import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Partitions [3, 2]
    n_parts = [3, 2]
    m_parts = [3, 2]
    p_parts = [3, 2]
    
    n_starts = [0, 3]
    m_starts = [0, 3]
    p_starts = [0, 3]
    
    # Calculate total rank
    # 3x3x3: 27 (Schoolbook)
    # 2x2x2: 7 (Strassen)
    # 3x3x2: 18
    # 3x2x3: 18
    # 3x2x2: 12
    # 2x3x3: 18
    # 2x3x2: 12
    # 2x2x3: 12
    total_rank = 27 + 7 + 18 + 18 + 12 + 18 + 12 + 12
    # 124? 
    # 3x3x3: 27
    # 3x3x2: 3*3*2 = 18
    # 3x2x3: 3*2*3 = 18
    # 3x2x2: 3*2*2 = 12
    # 2x3x3: 2*3*3 = 18
    # 2x3x2: 2*3*2 = 12
    # 2x2x3: 2*2*3 = 12
    # 2x2x2: 7
    # Sum: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124.
    
    # Wait, previous estimate was 70. Why the difference?
    # Ah, I assumed mixed blocks were 12 each.
    # 3x2x2 is 12.
    # 2x3x2 is 12.
    # 2x2x3 is 12.
    # But there are 3x3x2, 3x2x3, 2x3x3 blocks too!
    # 3x3x2: 3*3*2 = 18.
    # 3x2x3: 3*2*3 = 18.
    # 2x3x3: 2*3*3 = 18.
    # So yes, 124 is correct for schoolbook mixed.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen vectors for 2x2x2
    # Local indices: U (2i+j), V (2j+k), W (2k+i)
    S_U = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [-1, 0, 1, 0],# 6
        [0, 1, 0, -1] # 7
    ]
    S_V = [
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 1, 0, -1],# 3
        [-1, 0, 1, 0],# 4
        [0, 0, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ]
    S_W = [
        [1, 0, 0, 1], # 1
        [0, 1, 0, 0], # 2
        [0, 0, 1, 0], # 3
        [1, 1, 0, 0], # 4
        [-1, 0, 1, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 0, 0, 0]  # 7
    ]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    for r in range(7):
                        # U
                        for lu in range(4):
                            if S_U[r][lu] != 0:
                                li = lu // 2
                                lj = lu % 2
                                gu = (i_start + li) * m + (j_start + lj)
                                U[gu][col] += S_U[r][lu]
                        # V
                        for lv in range(4):
                            if S_V[r][lv] != 0:
                                lj = lv // 2
                                lk = lv % 2
                                gv = (j_start + lj) * p + (k_start + lk)
                                V[gv][col] += S_V[r][lv]
                        # W
                        for lw in range(4):
                            if S_W[r][lw] != 0:
                                lk = lw // 2
                                li = lw % 2
                                gw = (k_start + lk) * n + (i_start + li)
                                W[gw][col] += S_W[r][lw]
                        col += 1
                else:
                    # Schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
