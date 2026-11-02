import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    
    # Strassen factors for 2x2x2
    # U_core: 4x7, V_core: 4x7, W_core: 4x7
    # Indices:
    # U row: i*m + j
    # V row: j*p_local + k_local -> j*2 + r (since local p is 2)
    # W row: k_local*n + i -> r*2 + i
    
    # Strassen definitions
    # M1: (a00+a11)(b00+b11) -> c00, c11
    # M2: (a10+a11)b00 -> c10, -c11
    # M3: a00(b01-b11) -> c01, c11
    # M4: a11(b10-b00) -> c00, c10
    # M5: (a00+a01)b11 -> -c00, c01
    # M6: (a10-a00)(b00+b01) -> c11
    # M7: (a01-a11)(b10+b11) -> c00
    
    # U_core entries (i, j, coeff)
    u_entries = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    # V_core entries (j, r, coeff) where r is 0 or 1 (local col)
    v_entries = [
        [(0,0,1), (1,1,1)], # M1: b00, b11
        [(0,0,1)],          # M2: b00
        [(0,1,1), (1,1,-1)],# M3: b01, -b11
        [(1,0,1), (0,0,-1)],# M4: b10, -b00
        [(1,1,1)],          # M5: b11
        [(0,0,1), (0,1,1)], # M6: b00, b01
        [(1,0,1), (1,1,1)]  # M7: b10, b11
    ]
    
    # W_core entries (r, i, coeff) where r is 0 or 1 (local col), i is 0 or 1 (row)
    # Contribution to C[i, r]
    w_entries = [
        [(0,0,1), (1,1,1)], # M1: c00, c11
        [(1,0,1), (1,1,-1)],# M2: c10, -c11
        [(0,1,1), (1,1,1)], # M3: c01, c11
        [(0,0,1), (1,0,1)], # M4: c00, c10
        [(0,0,-1), (0,1,1)],# M5: -c00, c01
        [(1,1,1)],          # M6: c11
        [(0,0,1)]           # M7: c00
    ]
    
    # Build core matrices (4x7)
    U_core = [[0]*7 for _ in range(4)]
    V_core = [[0]*7 for _ in range(4)]
    W_core = [[0]*7 for _ in range(4)]
    
    for s in range(7):
        for (i, j, c) in u_entries[s]:
            U_core[i*2+j][s] = c
        for (j, r, c) in v_entries[s]:
            V_core[j*2+r][s] = c
        for (r, i, c) in w_entries[s]:
            W_core[r*2+i][s] = c
            
    # Expand to full size
    # U: 4 x 56. Repeat U_core 8 times horizontally.
    U = np.tile(U_core, (1, num_blocks)).tolist()
    
    # V: 32 x 56.
    # Row index in V: j*16 + k. k = 2t + r.
    # We want to place V_core block for t at rows corresponding to k=2t, 2t+1 for j=0,1.
    # Rows for block t: 
    # j=0, r=0 -> k=2t -> row 0*16 + 2t = 2t
    # j=0, r=1 -> k=2t+1 -> row 0*16 + 2t+1 = 2t+1
    # j=1, r=0 -> k=2t -> row 1*16 + 2t = 16+2t
    # j=1, r=1 -> k=2t+1 -> row 1*16 + 2t+1 = 16+2t+1
    # So for each t, the 4 rows are [2t, 2t+1, 16+2t, 16+2t+1].
    # The 7 columns for block t are [7t, ..., 7t+6].
    
    V = [[0]*56 for _ in range(32)]
    for t in range(num_blocks):
        cols = slice(7*t, 7*t+7)
        rows = [2*t, 2*t+1, 16+2*t, 16+2*t+1]
        for i, row_idx in enumerate(rows):
            V[row_idx][cols] = V_core[i]
            
    # W: 32 x 56.
    # Row index in W: k*n + i = (2t+r)*2 + i = 4t + 2r + i.
    # Rows for block t:
    # r=0, i=0 -> 4t
    # r=0, i=1 -> 4t+1
    # r=1, i=0 -> 4t+2
    # r=1, i=1 -> 4t+3
    # So rows are [4t, 4t+1, 4t+2, 4t+3].
    # Columns are [7t, ..., 7t+6].
    
    W = [[0]*56 for _ in range(32)]
    for t in range(num_blocks):
        cols = slice(7*t, 7*t+7)
        rows = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for i, row_idx in enumerate(rows):
            W[row_idx][cols] = W_core[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
