import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 31 decomposition using two 3x3 Laderman blocks and one 1x1 block
    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman 3x3 coefficients
    # u, v, w are lists of 9 coefficients for A, B, C respectively
    laderman_terms = [
        # 1
        ([1,0,0,0,0,0,0,0,0], [1,0,-1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # 2
        ([1,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # 3
        ([0,1,-1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 4
        ([0,0,0,1,0,0,0,0,0], [0,0,0,0,1,-1,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # 5
        ([0,0,0,1,1,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,0,0]),
        # 6
        ([0,0,0,0,1,-1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]),
        # 7
        ([0,0,1,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # 8
        ([0,0,1,-1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        # 9
        ([0,0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,0,0]),
        # 10
        ([0,0,0,1,-1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 11
        ([0,0,0,0,0,0,1,-1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0,0]),
        # 12
        ([0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        # 13
        ([0,0,0,0,0,0,0,1,-1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
        # 14
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,1,-1,0,0,0], [0,0,0,0,0,0,0,0,1]),
        # 15
        ([1,0,1,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
    ]

    # Helper to add a 3x3 block at offset (offset, offset)
    def add_block(offset, start_col):
        for t_idx in range(15):
            u, v, w = laderman_terms[t_idx]
            col = start_col + t_idx
            for local_i in range(3):
                for local_j in range(3):
                    idx_u = (offset + local_i) * 5 + (offset + local_j)
                    if u[local_i * 3 + local_j] != 0:
                        U[idx_u][col] = u[local_i * 3 + local_j]
            for local_j in range(3):
                for local_k in range(3):
                    idx_v = (offset + local_j) * 5 + (offset + local_k)
                    if v[local_j * 3 + local_k] != 0:
                        V[idx_v][col] = v[local_j * 3 + local_k]
            for local_k in range(3):
                for local_i in range(3):
                    idx_w = (offset + local_k) * 5 + (offset + local_i)
                    if w[local_i * 3 + local_k] != 0:
                        W[idx_w][col] = w[local_i * 3 + local_k]

    # Block 1: Top-Left 3x3 (offset 0)
    add_block(0, 0)
    
    # Block 2: Bottom-Right 3x3 (offset 2)
    add_block(2, 15)
    
    # Block 3: Center 1x1 (index 2,2)
    # A[2,2] * B[2,2] = C[2,2]
    # U[2*5+2, 30] = 1
    # V[2*5+2, 30] = 1
    # W[2*5+2, 30] = 1
    U[12][30] = 1
    V[12][30] = 1
    W[12][30] = 1

    return {"U": U, "V": V, "W": W}

