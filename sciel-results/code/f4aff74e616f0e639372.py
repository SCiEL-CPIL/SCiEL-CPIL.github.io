"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition
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
        [0, 0, 1, -1],
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
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_offset in range(2):
        for s in range(7):
            for t in range(7):
                col_idx = block_offset * 49 + s * 7 + t
                
                for bi in range(2):
                    for bj in range(2):
                        outer_u = strassen_U[s][bi * 2 + bj]
                        outer_v = strassen_V[s][bi * 2 + bj]
                        outer_w = strassen_W[s][bi * 2 + bj]
                        
                        for ii in range(2):
                            for jj in range(2):
                                inner_u = strassen_U[t][ii * 2 + jj]
                                inner_v = strassen_V[t][ii * 2 + jj]
                                inner_w = strassen_W[t][ii * 2 + jj]
                                
                                if outer_u != 0 and inner_u != 0:
                                    idx = (2*bi + ii) * 4 + (2*bj + jj)
                                    U[idx][col_idx] += outer_u * inner_u
                                
                                if outer_v != 0 and inner_v != 0:
                                    row = 2*bi + ii
                                    col = 4*block_offset + 2*bj + jj
                                    idx = row * 8 + col
                                    V[idx][col_idx] += outer_v * inner_v
                                
                                if outer_w != 0 and inner_w != 0:
                                    i_idx = 2*bi + ii
                                    k_idx = 4*block_offset + 2*bj + jj
                                    idx = k_idx * 4 + i_idx
                                    W[idx][col_idx] += outer_w * inner_w
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
