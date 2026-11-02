import numpy as np

def construct():
    m = 2
    n = 2
    p = 16
    num_blocks = p // n  # 8 blocks
    block_rank = 7
    rank = num_blocks * block_rank  # 56
    
    U = np.zeros((m * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((p * m, rank))
    
    # Strassen coefficients
    # A indices: 0: a, 1: b, 2: c, 3: d
    # L_A for each term
    L_A = [
        [1, 0, 0, 1],  # m1: a+d
        [0, 0, 1, 1],  # m2: c+d
        [1, 0, 0, 0],  # m3: a
        [0, 0, 0, 1],  # m4: d
        [1, 1, 0, 0],  # m5: a+b
        [-1, 0, 1, 0], # m6: c-a
        [0, 1, 0, -1]  # m7: b-d
    ]
    
    # B indices within block: 0: e, 1: f, 2: g, 3: h
    # Global B index = j * p + l. 
    # Block k starts at col 2k.
    # e: j=0, l=2k -> 0*p + 2k = 2k
    # f: j=0, l=2k+1 -> 2k+1
    # g: j=1, l=2k -> p + 2k
    # h: j=1, l=2k+1 -> p + 2k+1
    # L_B for each term (coeffs for e, f, g, h)
    L_B = [
        [1, 0, 0, 1],  # m1: e+h
        [1, 0, 0, 0],  # m2: e
        [0, 1, 0, -1], # m3: f-h
        [0, 0, 1, -1], # m4: g-e
        [0, 0, 0, 1],  # m5: h
        [1, 0, 1, 0],  # m6: e+g
        [0, 0, 1, 1]   # m7: g+h
    ]
    
    # C indices within block: 0: p, 1: q, 2: r, 3: s
    # Global C index = i * p + l.
    # Block k starts at col 2k.
    # p: i=0, l=2k -> 2k
    # q: i=0, l=2k+1 -> 2k+1
    # r: i=1, l=2k -> p + 2k
    # s: i=1, l=2k+1 -> p + 2k+1
    # L_C for each term (coeffs for p, q, r, s)
    # Reconstruction:
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    L_C = [
        [1, 0, 0, 1],  # m1: p+s
        [0, 0, 1, -1], # m2: r-s
        [0, 1, 0, 1],  # m3: q+s
        [1, 0, 1, 0],  # m4: p+r
        [0, 1, 0, 0],  # m5: q (wait, p gets -1) -> [ -1, 1, 0, 0 ]
        [0, 0, 0, 1],  # m6: s
        [1, 0, 0, 0]   # m7: p
    ]
    # Correction for m5 L_C:
    # p = ... - m5 ... => coeff -1 for p
    # q = ... + m5 ... => coeff 1 for q
    # So L_C[4] = [-1, 1, 0, 0]
    L_C[4] = [-1, 1, 0, 0]
    
    for k in range(num_blocks):
        for r in range(block_rank):
            idx = k * block_rank + r
            
            # Fill U
            for i in range(4):
                U[i, idx] = L_A[r][i]
                
            # Fill V
            # e: 2k, f: 2k+1, g: p+2k, h: p+2k+1
            V[2*k, idx] = L_B[r][0]
            V[2*k+1, idx] = L_B[r][1]
            V[p + 2*k, idx] = L_B[r][2]
            V[p + 2*k + 1, idx] = L_B[r][3]
            
            # Fill W
            # p: 2k, q: 2k+1, r: p+2k, s: p+2k+1
            W[2*k, idx] = L_C[r][0]
            W[2*k+1, idx] = L_C[r][1]
            W[p + 2*k, idx] = L_C[r][2]
            W[p + 2*k + 1, idx] = L_C[r][3]
            
    return {'U': U, 'V': V, 'W': W}
