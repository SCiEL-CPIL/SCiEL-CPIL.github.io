import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49
    rank = 49
    
    # U: (n*m, rank) = (4, 49)
    # V: (m*p, rank) = (28, 49)
    # W: (p*n, rank) = (28, 49)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (indices 0:00, 1:01, 2:10, 3:11)
    # M1: a+d -> [1, 0, 0, 1]
    # M2: c+d -> [0, 0, 1, 1]
    # M3: a   -> [1, 0, 0, 0]
    # M4: d   -> [0, 0, 0, 1]
    # M5: a+b -> [1, 1, 0, 0]
    # M6: c-a -> [-1, 0, 1, 0]
    # M7: b-d -> [0, 1, 0, -1]
    
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # For each block k (0..6), B is 2x2:
    # e = B[0, 2k]
    # f = B[0, 2k+1]
    # g = B[1, 2k]
    # h = B[1, 2k+1]
    
    # V coefficients for each M (indices for B flattened j*p + k)
    # p=14.
    # e: idx 0*14 + 2k
    # f: idx 0*14 + 2k+1
    # g: idx 1*14 + 2k
    # h: idx 1*14 + 2k+1
    
    # M1: e+h -> {e:1, h:1}
    # M2: e   -> {e:1}
    # M3: f-h -> {f:1, h:-1}
    # M4: g-e -> {g:1, e:-1}
    # M5: h   -> {h:1}
    # M6: e+f -> {e:1, f:1}
    # M7: g+h -> {g:1, h:1}
    
    # W coefficients for each M (indices for C flattened k*n + i)
    # p=14, n=2.
    # C indices:
    # p_out = C[0, 2k]   -> idx 2k*2 + 0
    # q_out = C[0, 2k+1] -> idx (2k+1)*2 + 0
    # r_out = C[1, 2k]   -> idx 2k*2 + 1
    # s_out = C[1, 2k+1] -> idx (2k+1)*2 + 1
    
    # M1: p:1, s:1
    # M2: r:1, s:-1
    # M3: q:1, s:1
    # M4: p:1, r:1
    # M5: p:-1, q:1
    # M6: s:1
    # M7: p:1
    
    # We will fill U, V, W
    
    for k in range(7):
        # B indices
        idx_e = 0 * p + 2 * k
        idx_f = 0 * p + 2 * k + 1
        idx_g = 1 * p + 2 * k
        idx_h = 1 * p + 2 * k + 1
        
        # C indices
        idx_p = (2 * k) * n + 0
        idx_q = (2 * k + 1) * n + 0
        idx_r = (2 * k) * n + 1
        idx_s = (2 * k + 1) * n + 1
        
        # V definitions (list of (index, coeff))
        V_defs = [
            [(idx_e, 1), (idx_h, 1)], # M1
            [(idx_e, 1)],             # M2
            [(idx_f, 1), (idx_h, -1)],# M3
            [(idx_g, 1), (idx_e, -1)],# M4
            [(idx_h, 1)],             # M5
            [(idx_e, 1), (idx_f, 1)], # M6
            [(idx_g, 1), (idx_h, 1)]  # M7
        ]
        
        # W definitions (list of (index, coeff))
        W_defs = [
            [(idx_p, 1), (idx_s, 1)], # M1
            [(idx_r, 1), (idx_s, -1)],# M2
            [(idx_q, 1), (idx_s, 1)], # M3
            [(idx_p, 1), (idx_r, 1)], # M4
            [(idx_p, -1), (idx_q, 1)],# M5
            [(idx_s, 1)],             # M6
            [(idx_p, 1)]              # M7
        ]
        
        for m_idx in range(7):
            r = 7 * k + m_idx
            
            # Fill U
            for i_idx in range(4):
                U[i_idx][r] = A_coeffs[m_idx][i_idx]
                
            # Fill V
            for v_idx, coeff in V_defs[m_idx]:
                V[v_idx][r] += coeff
                
            # Fill W
            for w_idx, coeff in W_defs[m_idx]:
                W[w_idx][r] += coeff
                
    return {"U": U, "V": V, "W": W}
