import time
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 135.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Strassen 2x2 decomposition
    # Local indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U factors for 7 products
    # P1: a00 + a11 -> {0:1, 3:1}
    # P2: a10 + a11 -> {2:1, 3:1}
    # P3: a00 -> {0:1}
    # P4: a11 -> {3:1}
    # P5: a00 + a01 -> {0:1, 1:1}
    # P6: a10 - a00 -> {2:1, 0:-1}
    # P7: a01 - a11 -> {1:1, 3:-1}
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    # V factors for 7 products
    # P1: b00 + b11 -> {0:1, 3:1}
    # P2: b00 -> {0:1}
    # P3: b01 - b11 -> {1:1, 3:-1}
    # P4: b10 - b00 -> {2:1, 0:-1}
    # P5: b11 -> {3:1}
    # P6: b00 + b01 -> {0:1, 1:1}
    # P7: b10 + b11 -> {2:1, 3:1}
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # W factors for 7 products
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    #
    # P1: c00(+1), c11(+1) -> {0:1, 3:1}
    # P2: c10(+1), c11(-1) -> {2:1, 3:-1}
    # P3: c01(+1), c11(+1) -> {1:1, 3:1}
    # P4: c00(+1), c10(+1) -> {0:1, 2:1}
    # P5: c00(-1), c01(+1) -> {0:-1, 1:1}
    # P6: c11(+1) -> {3:1}
    # P7: c00(+1) -> {0:1}
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    rank_total = 42
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]
    
    col = 0
    for m_block in range(3):
        # Group 1: A_T (cols 0,1) * B_T (rows 0,1)
        # A local (u,v) -> global (u, v)
        # B local (v,w) -> global (v, 2*m_block + w)
        # C local (u,w) -> global (u, 2*m_block + w)
        for prod_idx in range(7):
            # U
            for local_idx, val in strassen_U[prod_idx].items():
                u_local, v_local = local_idx // 2, local_idx % 2
                global_i = u_local
                global_j = v_local
                U[global_i * m + global_j, col] += val
            
            # V
            for local_idx, val in strassen_V[prod_idx].items():
                v_local, w_local = local_idx // 2, local_idx % 2
                global_j = v_local
                global_k = 2 * m_block + w_local
                V[global_j * p + global_k, col] += val
            
            # W
            for local_idx, val in strassen_W[prod_idx].items():
                u_local, w_local = local_idx // 2, local_idx % 2
                global_i = u_local
                global_k = 2 * m_block + w_local
                W[global_k * n + global_i, col] += val
            
            col += 1

        # Group 2: A_B (cols 2,3) * B_B (rows 2,3)
        # A local (u,v) -> global (u, 2 + v)
        # B local (v,w) -> global (2 + v, 2*m_block + w)
        # C local (u,w) -> global (u, 2*m_block + w)
        for prod_idx in range(7):
            # U
            for local_idx, val in strassen_U[prod_idx].items():
                u_local, v_local = local_idx // 2, local_idx % 2
                global_i = u_local
                global_j = 2 + v_local
                U[global_i * m + global_j, col] += val
            
            # V
            for local_idx, val in strassen_V[prod_idx].items():
                v_local, w_local = local_idx // 2, local_idx % 2
                global_j = 2 + v_local
                global_k = 2 * m_block + w_local
                V[global_j * p + global_k, col] += val
            
            # W
            for local_idx, val in strassen_W[prod_idx].items():
                u_local, w_local = local_idx // 2, local_idx % 2
                global_i = u_local
                global_k = 2 * m_block + w_local
                W[global_k * n + global_i, col] += val
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
