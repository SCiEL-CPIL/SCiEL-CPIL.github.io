import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for A: 0=a, 1=b, 2=c, 3=d
    # Local indices for B: 0=e, 1=f, 2=g, 3=h
    
    # U coefficients for each M
    # M1: a, d
    # M2: c, d
    # M3: a
    # M4: d
    # M5: a, b
    # M6: c, -a
    # M7: b, -d
    
    # V coefficients for each M
    # M1: e, h
    # M2: e
    # M3: f, -h
    # M4: g, -e
    # M5: h
    # M6: e, f
    # M7: g, h
    
    # W coefficients for each P
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M4
    # P22 = M1 - M2 + M3 + M6
    
    # We define the blocks
    # Block k: 0..5
    # For each block, we define which global A indices, B indices, and C indices are involved.
    
    # A indices:
    # A1 (cols 0,1): [0, 1, 4, 5] -> a,b,c,d
    # A2 (cols 2,3): [2, 3, 6, 7] -> a,b,c,d
    
    # B indices:
    # B1 (rows 0,1, cols 0,1): [0, 1, 6, 7] -> e,f,g,h
    # B2 (rows 0,1, cols 2,3): [2, 3, 8, 9]
    # B3 (rows 0,1, cols 4,5): [4, 5, 10, 11]
    # B4 (rows 2,3, cols 0,1): [12, 13, 18, 19]
    # B5 (rows 2,3, cols 2,3): [14, 15, 20, 21]
    # B6 (rows 2,3, cols 4,5): [16, 17, 22, 23]
    
    # C indices (output):
    # C1 (cols 0,1): [0, 1, 2, 3] -> P11, P12, P21, P22
    # C2 (cols 2,3): [4, 5, 6, 7]
    # C3 (cols 4,5): [8, 9, 10, 11]
    
    # Mapping:
    # Block 0: A1, B1, C1
    # Block 1: A1, B2, C2
    # Block 2: A1, B3, C3
    # Block 3: A2, B4, C1 (adds to C1)
    # Block 4: A2, B5, C2 (adds to C2)
    # Block 5: A2, B6, C3 (adds to C3)
    
    blocks = [
        # (A_indices, B_indices, C_indices)
        ([0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 2, 3]),
        ([0, 1, 4, 5], [2, 3, 8, 9], [4, 5, 6, 7]),
        ([0, 1, 4, 5], [4, 5, 10, 11], [8, 9, 10, 11]),
        ([2, 3, 6, 7], [12, 13, 18, 19], [0, 1, 2, 3]),
        ([2, 3, 6, 7], [14, 15, 20, 21], [4, 5, 6, 7]),
        ([2, 3, 6, 7], [16, 17, 22, 23], [8, 9, 10, 11]),
    ]
    
    for b_idx, (a_idx, b_idx_list, c_idx) in enumerate(blocks):
        base_col = b_idx * 7
        for m_idx in range(7):
            col = base_col + m_idx
            if m_idx == 0: # M1
                U[a_idx[0]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx_list[0]][col] += 1
                V[b_idx_list[3]][col] += 1
            elif m_idx == 1: # M2
                U[a_idx[2]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx_list[0]][col] += 1
            elif m_idx == 2: # M3
                U[a_idx[0]][col] += 1
                V[b_idx_list[1]][col] += 1
                V[b_idx_list[3]][col] += -1
            elif m_idx == 3: # M4
                U[a_idx[3]][col] += 1
                V[b_idx_list[2]][col] += 1
                V[b_idx_list[0]][col] += -1
            elif m_idx == 4: # M5
                U[a_idx[0]][col] += 1
                U[a_idx[1]][col] += 1
                V[b_idx_list[3]][col] += 1
            elif m_idx == 5: # M6
                U[a_idx[2]][col] += 1
                U[a_idx[0]][col] += -1
                V[b_idx_list[0]][col] += 1
                V[b_idx_list[1]][col] += 1
            elif m_idx == 6: # M7
                U[a_idx[1]][col] += 1
                U[a_idx[3]][col] += -1
                V[b_idx_list[2]][col] += 1
                V[b_idx_list[3]][col] += 1
            
            # Update W
            if m_idx == 0: # M1
                W[c_idx[0]][col] += 1 # P11
                W[c_idx[3]][col] += 1 # P22
            elif m_idx == 1: # M2
                W[c_idx[2]][col] += 1 # P21
                W[c_idx[3]][col] += -1 # P22
            elif m_idx == 2: # M3
                W[c_idx[1]][col] += 1 # P12
                W[c_idx[3]][col] += 1 # P22
            elif m_idx == 3: # M4
                W[c_idx[0]][col] += 1 # P11
                W[c_idx[2]][col] += 1 # P21
            elif m_idx == 4: # M5
                W[c_idx[0]][col] += -1 # P11
                W[c_idx[1]][col] += 1 # P12
            elif m_idx == 5: # M6
                W[c_idx[3]][col] += 1 # P22
            elif m_idx == 6: # M7
                W[c_idx[0]][col] += 1 # P11

    return {"U": U, "V": V, "W": W}
