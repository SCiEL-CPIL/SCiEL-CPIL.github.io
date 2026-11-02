import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 Strassen decomposition (rank 49)
    
    def get_vec(blocks):
        vec = [0.0] * 16
        for block_id, coeff in blocks:
            r0 = (block_id // 2) * 2
            c0 = (block_id % 2) * 2
            for dr in range(2):
                for dc in range(2):
                    idx = (r0 + dr) * 4 + (c0 + dc)
                    vec[idx] += coeff
        return vec
    
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, -1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 0, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 0, 1], [-1, 1, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 1, -1, 0]),
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 1])
    ]
    
    M_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(3, 1), (0, -1)]),
        ([(3, 1)], [(3, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(0, 1), (2, 1)])
    ]
    
    col = 0
    for i in range(7):
        if time.time() > deadline:
            break
        A_blocks, B_blocks = M_defs[i]
        A_vec = get_vec(A_blocks)
        B_vec = get_vec(B_blocks)
        for k in range(7):
            u_k, v_k, w_k = S[k]
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            for idx in range(16):
                r = idx // 4
                c = idx % 4
                pos = (r % 2) * 2 + (c % 2)
                U_vec[idx] = A_vec[idx] * u_k[pos]
                V_vec[idx] = B_vec[idx] * v_k[pos]
                W_vec[idx] = w_k[pos]
            for idx in range(16):
                U[idx][col] += U_vec[idx]
                V[idx][col] += V_vec[idx]
                W[idx][col] += W_vec[idx]
            col += 1
            
    # Part 2: 4x4x3 Schoolbook decomposition (rank 48)
    for i in range(4):
        if time.time() > deadline:
            break
        for j in range(4):
            for k in range(3):
                kk = k + 4
                U[i * 4 + j][col] = 1
                V[j * 7 + kk][col] = 1
                W[kk * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
