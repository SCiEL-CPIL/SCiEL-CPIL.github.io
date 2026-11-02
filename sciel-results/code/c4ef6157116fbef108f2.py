import time

DIMENSIONS = [2, 2, 14]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen terms for 2x2 multiplication
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local indices per block: 0=B00, 1=B01, 2=B10, 3=B11
    # W local indices per block: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # V global offsets: B00->2b, B01->2b+1, B10->14+2b, B11->14+2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        # W global offsets: C00->4b, C01->4b+2, C10->4b+1, C11->4b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
        
        # Time guard
        if time.time() - start_time > 100:
            break
    
    return {"U": U, "V": V, "W": W}
