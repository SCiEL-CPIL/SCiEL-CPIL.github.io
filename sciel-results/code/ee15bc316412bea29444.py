import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices for A (2x2): 00, 01, 10, 11 -> u_idx
    # Local indices for B (2x2): 00, 01, 10, 11 -> v_idx
    # Local indices for C (2x2): 00, 01, 10, 11 -> w_idx
    
    # Mappings for U (A part)
    # A00: u=0, A01: u=1, A10: u=2, A11: u=3
    # Mappings for V (B part)
    # B00: v=0, B01: v=1, B10: v=2, B11: v=3
    # Mappings for W (C part)
    # C00: w=0, C01: w=1, C10: w=2, C11: w=3
    
    # Strassen terms (coeff, u_indices, v_indices, w_indices)
    # u_indices is list of (local_u, coeff)
    # v_indices is list of (local_v, coeff)
    # w_indices is list of (local_w, coeff)
    
    terms = [
        # M1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)] # C00 gets +1, C11 gets +1
        },
        # M2
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)] # C10 gets +1, C11 gets -1
        },
        # M3
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)] # C01 gets +1, C11 gets +1
        },
        # M4
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)] # C00 gets +1, C10 gets +1
        },
        # M5
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)] # C00 gets -1, C01 gets +1
        },
        # M6
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)] # C11 gets +1
        },
        # M7
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)] # C00 gets +1
        }
    ]
    
    # Define the 4 blocks
    # Block 1: P1 = A1 B11 -> C_left (cols 0-1)
    # A1: global u = i*4 + j_local. i in 0..1, j_local in 0..1.
    # B11: global v = j_row*8 + k_local. j_row in 0..1, k_local in 0..1.
    # C: global w = k_global*2 + i. k_global in 0..1, i in 0..1.
    
    # Block 2: P2 = A2 B21 -> C_left (cols 0-1)
    # A2: global u = i*4 + 2 + j_local.
    # B21: global v = (2+j_row)*8 + k_local.
    # C: global w = k_global*2 + i.
    
    # Block 3: P3 = A1 B12 -> C_right (cols 2-3)
    # A1: global u = i*4 + j_local.
    # B12: global v = j_row*8 + 2 + k_local.
    # C: global w = (k_global+2)*2 + i = 2*k_global + 4 + i.
    
    # Block 4: P4 = A2 B22 -> C_right (cols 2-3)
    # A2: global u = i*4 + 2 + j_local.
    # B22: global v = (2+j_row)*8 + 2 + k_local.
    # C: global w = (k_global+2)*2 + i.

    col = 0
    for block_idx in range(4):
        for term in terms:
            if time.time() > deadline:
                break
            # Determine offsets
            if block_idx == 0: # P1
                u_offset = 0
                v_offset = 0
                w_offset = 0
            elif block_idx == 1: # P2
                u_offset = 2
                v_offset = 16 # 2*8
                w_offset = 0
            elif block_idx == 2: # P3
                u_offset = 0
                v_offset = 2
                w_offset = 4 # 2*2
            elif block_idx == 3: # P4
                u_offset = 2
                v_offset = 18 # 2*8 + 2
                w_offset = 4
            
            # Fill U
            for lu, coeff in term['u']:
                # lu: 0,1,2,3 -> i, j_local
                i = lu // 2
                jl = lu % 2
                gu = i * 4 + jl + u_offset
                U[gu][col] += coeff
            
            # Fill V
            for lv, coeff in term['v']:
                # lv: 0,1,2,3 -> j_row, k_local
                jr = lv // 2
                kl = lv % 2
                gv = jr * 8 + kl + v_offset
                V[gv][col] += coeff
            
            # Fill W
            for lw, coeff in term['w']:
                # lw: 0,1,2,3 -> k_global, i
                kg = lw // 2
                i = lw % 2
                gw = kg * 2 + i + w_offset
                W[gw][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
