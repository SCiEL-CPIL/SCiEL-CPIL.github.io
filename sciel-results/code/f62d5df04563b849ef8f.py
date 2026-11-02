import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for A (u), B (v), C (w)
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (per block): 00, 01, 10, 11
    # Indices for C (per block): 00, 01, 10, 11
    
    # u_r for r=0..6
    u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # v_coeffs for r=0..6 (relative to block B)
    # B indices in block: 00, 01, 10, 11
    v_coeffs = [
        [1, 0, 0, 1], # M1: b00 + b11
        [1, 0, 0, 0], # M2: b00
        [0, 1, 0, -1],# M3: b01 - b11
        [-1, 0, 1, 0],# M4: b10 - b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00 + b01
        [0, 0, 1, 1]  # M7: b10 + b11
    ]
    
    # w_coeffs for r=0..6 (relative to block C)
    # C indices in block: 00, 01, 10, 11
    w_coeffs = [
        [1, 0, 0, 1], # M1: c00 += 1, c11 += 1
        [0, 0, 1, -1],# M2: c10 += 1, c11 += -1
        [0, 1, 0, 1], # M3: c01 += 1, c11 += 1
        [1, 0, 1, 0], # M4: c00 += 1, c10 += 1
        [-1, 1, 0, 0],# M5: c00 += -1, c01 += 1
        [0, 0, 0, 0], # M6: not used?
        [0, -1, 0, 1] # M7: c01 += -1, c11 += 1
    ]
    
    # Fill U, V, W
    col = 0
    for b in range(3):
        for r in range(7):
            # U
            for i in range(4):
                U[i, col] = u_coeffs[r][i]
            
            # V
            # Block b of B
            # B flat indices: j*p + k
            # j=0, k=2b -> 2b
            # j=0, k=2b+1 -> 2b+1
            # j=1, k=2b -> 6+2b
            # j=1, k=2b+1 -> 6+2b+1
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for i in range(4):
                V[v_idx[i], col] = v_coeffs[r][i]
            
            # W
            # Block b of C
            # C flat indices: k*n + i
            # i=0, k=2b -> 4b
            # i=1, k=2b -> 4b+1
            # i=0, k=2b+1 -> 4b+2
            # i=1, k=2b+1 -> 4b+3
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for i in range(4):
                W[w_idx[i], col] = w_coeffs[r][i]
                
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
