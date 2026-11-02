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
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for A (indices 0:00, 1:01, 2:10, 3:11)
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
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
            U[:, r] = A_coeffs[m_idx]
            
            # Fill V
            for v_idx, coeff in V_defs[m_idx]:
                V[v_idx, r] += coeff
                
            # Fill W
            for w_idx, coeff in W_defs[m_idx]:
                W[w_idx, r] += coeff
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
