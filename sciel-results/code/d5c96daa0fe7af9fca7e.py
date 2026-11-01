import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # We use block Strassen.
    # p=14, so we have 7 blocks of width 2.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen coefficients for U (A matrix parts: a, b, c, d)
    # Indices in U: 0=a, 1=b, 2=c, 3=d
    U_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 0, 1, -1], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # Strassen coefficients for V (B matrix parts: e, f, g, h)
    # Local indices 0=e, 1=f, 2=g, 3=h
    V_coeffs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # Strassen contributions to W (C matrix parts)
    # C indices: 0=C00, 1=C10, 2=C01, 3=C11
    # Note: W row index for C[i, j] is j*n + i.
    # For block k, columns are 2k and 2k+1.
    # C00 (i=0, j=2k) -> W[2k*2 + 0] = W[4k]
    # C10 (i=1, j=2k) -> W[2k*2 + 1] = W[4k+1]
    # C01 (i=0, j=2k+1) -> W[(2k+1)*2 + 0] = W[4k+2]
    # C11 (i=1, j=2k+1) -> W[(2k+1)*2 + 1] = W[4k+3]
    
    W_coeffs = [
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 1, 0, -1], # M2: C10+1, C11-1
        [0, 0, 1, 1],  # M3: C01+1, C11+1
        [1, 1, 0, 0],  # M4: C00+1, C10+1
        [-1, 0, 1, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0]   # M7: C00+1
    ]
    
    col = 0
    for k in range(num_blocks):
        # V indices for this block
        # e = B[0, 2k] -> 2k
        # f = B[0, 2k+1] -> 2k+1
        # g = B[1, 2k] -> 14 + 2k
        # h = B[1, 2k+1] -> 14 + 2k+1
        v_idx = [2*k, 2*k+1, 14+2*k, 14+2*k+1]
        
        # W indices for this block
        w_idx = [4*k, 4*k+1, 4*k+2, 4*k+3]
        
        for m_idx in range(rank_per_block):
            # Set U
            U[0, col] = U_coeffs[m_idx][0]
            U[1, col] = U_coeffs[m_idx][1]
            U[2, col] = U_coeffs[m_idx][2]
            U[3, col] = U_coeffs[m_idx][3]
            
            # Set V
            V[v_idx[0], col] = V_coeffs[m_idx][0]
            V[v_idx[1], col] = V_coeffs[m_idx][1]
            V[v_idx[2], col] = V_coeffs[m_idx][2]
            V[v_idx[3], col] = V_coeffs[m_idx][3]
            
            # Set W
            W[w_idx[0], col] = W_coeffs[m_idx][0]
            W[w_idx[1], col] = W_coeffs[m_idx][1]
            W[w_idx[2], col] = W_coeffs[m_idx][2]
            W[w_idx[3], col] = W_coeffs[m_idx][3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
