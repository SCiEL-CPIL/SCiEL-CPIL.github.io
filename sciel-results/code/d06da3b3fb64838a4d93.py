import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    
    # Core Strassen decomposition for 2x2x2
    # U_core: 4x7, V_core: 4x7, W_core: 4x7
    # Indices for A (4), B (4), C (4)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_core = np.zeros((4, 7), dtype=int)
    V_core = np.zeros((4, 7), dtype=int)
    W_core = np.zeros((4, 7), dtype=int)
    
    # Strassen factors
    # s=0: M1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
    U_core[0, 0] = 1; U_core[3, 0] = 1
    V_core[0, 0] = 1; V_core[3, 0] = 1
    W_core[0, 0] = 1; W_core[3, 0] = 1
    
    # s=1: M2 = (a10+a11)b00 -> c10+=1, c11+=-1
    U_core[2, 1] = 1; U_core[3, 1] = 1
    V_core[0, 1] = 1
    W_core[2, 1] = 1; W_core[3, 1] = -1
    
    # s=2: M3 = a00(b01-b11) -> c01+=1, c11+=1
    U_core[0, 2] = 1
    V_core[1, 2] = 1; V_core[3, 2] = -1
    W_core[1, 2] = 1; W_core[3, 2] = 1
    
    # s=3: M4 = a11(b10-b00) -> c00+=1, c10+=1
    U_core[3, 3] = 1
    V_core[2, 3] = 1; V_core[0, 3] = -1
    W_core[0, 3] = 1; W_core[2, 3] = 1
    
    # s=4: M5 = (a00+a01)b11 -> c00+=-1, c01+=1
    U_core[0, 4] = 1; U_core[1, 4] = 1
    V_core[3, 4] = 1
    W_core[0, 4] = -1; W_core[1, 4] = 1
    
    # s=5: M6 = (a10-a00)(b00+b01) -> c11+=1
    U_core[2, 5] = 1; U_core[0, 5] = -1
    V_core[0, 5] = 1; V_core[1, 5] = 1
    W_core[3, 5] = 1
    
    # s=6: M7 = (a01-a11)(b10+b11) -> c00+=1
    U_core[1, 6] = 1; U_core[3, 6] = -1
    V_core[2, 6] = 1; V_core[3, 6] = 1
    W_core[0, 6] = 1

    rank = 7 * num_blocks
    
    # Construct U: 4 x 56
    # U is U_core repeated 8 times horizontally
    U = np.tile(U_core, (1, num_blocks))
    
    # Construct V: 32 x 56
    # V rows: j * 16 + k
    # For block t, k in {2t, 2t+1}
    # Rows used: j * 16 + 2t and j * 16 + 2t + 1 for j in {0, 1}
    # V_core rows: j * 2 + r for j in {0, 1}, r in {0, 1}
    # Mapping: V_core row (j, r) -> V row (j * 16 + 2t + r)
    
    V = np.zeros((m * p, rank), dtype=int)
    for t in range(num_blocks):
        for j in range(m):
            for r in range(2):
                v_row_core = j * 2 + r
                v_row_full = j * p + (2 * t + r)
                V[v_row_full, t * 7 : (t + 1) * 7] = V_core[v_row_core, :]
                
    # Construct W: 32 x 56
    # W rows: k * n + i
    # For block t, k in {2t, 2t+1}
    # Rows used: (2t + r) * n + i for r in {0, 1}, i in {0, 1}
    # W_core rows: r * 2 + i for r in {0, 1}, i in {0, 1}
    # Mapping: W_core row (r, i) -> W row ((2t + r) * n + i)
    
    W = np.zeros((p * n, rank), dtype=int)
    for t in range(num_blocks):
        for r in range(2):
            for i in range(n):
                w_row_core = r * 2 + i
                w_row_full = (2 * t + r) * n + i
                W[w_row_full, t * 7 : (t + 1) * 7] = W_core[w_row_core, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
