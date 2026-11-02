import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (known) tensor decomposition for 2x2x12 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks.
    Decomposes the 12 columns into 6 blocks of 2 columns.
    Each block requires rank 7.
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6 blocks
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block  # 42
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (24, 42)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[i, j], [k, l]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # Define the 7 terms as dictionaries of coefficients
    # Key: (index_in_matrix, value)
    # U indices: i*m + j  (0: a, 1: b, 2: c, 3: d)
    # V indices: j*p + k  (local k: 0,1)
    # W indices: k*n + i  (local k: 0,1)
    
    terms = [
        # M1
        {
            'U': {0: 1, 3: 1},  # a, d
            'V': {0: 1, 3: 1},  # e, h (local indices: 0*2+0=0, 1*2+1=3)
            'W': {0: 1, 3: 1}   # i, l (local indices: 0*2+0=0, 1*2+1=3)
        },
        # M2
        {
            'U': {2: 1, 3: 1},  # c, d
            'V': {0: 1},        # e
            'W': {2: 1, 3: -1}  # k, l (local: 0*2+1=2, 1*2+1=3)
        },
        # M3
        {
            'U': {0: 1},        # a
            'V': {1: 1, 3: -1}, # f, h (local: 0*2+1=1, 1*2+1=3)
            'W': {1: 1, 3: 1}   # j, l (local: 1*2+0=1, 1*2+1=3)
        },
        # M4
        {
            'U': {3: 1},        # d
            'V': {2: 1, 0: -1}, # g, e (local: 1*2+0=2, 0*2+0=0)
            'W': {0: 1, 2: 1}   # i, k (local: 0*2+0=0, 0*2+1=2)
        },
        # M5
        {
            'U': {0: 1, 1: 1},  # a, b
            'V': {3: 1},        # h
            'W': {0: -1, 1: 1}  # i, j (local: 0*2+0=0, 1*2+0=1)
        },
        # M6
        {
            'U': {2: 1, 0: -1}, # c, a
            'V': {0: 1, 1: 1},  # e, f
            'W': {3: 1}         # l
        },
        # M7
        {
            'U': {1: 1, 3: -1}, # b, d
            'V': {2: 1, 3: 1},  # g, h
            'W': {0: 1}         # i
        }
    ]
    
    # Fill the matrices
    for block_idx in range(num_blocks):
        base_rank = block_idx * rank_per_block
        # Global column offset for B and C
        col_offset = block_idx * 2
        
        for term_idx in range(rank_per_block):
            r = base_rank + term_idx
            term = terms[term_idx]
            
            # Fill U
            for u_idx, val in term['U'].items():
                U[u_idx][r] = val
                
            # Fill V
            for v_idx, val in term['V'].items():
                # v_idx is local (0..3)
                # Global row in V: j*p + k
                # Local v_idx corresponds to (j, k_local) where v_idx = j*2 + k_local
                j = v_idx // 2
                k_local = v_idx % 2
                k_global = col_offset + k_local
                v_row = j * p + k_global
                V[v_row][r] = val
                
            # Fill W
            for w_idx, val in term['W'].items():
                # w_idx is local (0..3)
                # Global row in W: k*n + i
                # Local w_idx corresponds to (k_local, i) where w_idx = k_local*2 + i
                k_local = w_idx // 2
                i = w_idx % 2
                k_global = col_offset + k_local
                w_row = k_global * n + i
                W[w_row][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
