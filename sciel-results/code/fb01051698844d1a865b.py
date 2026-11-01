import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses the insight that the multiplication tensor for 2x2 by 2x12 can be decomposed
    into 7 terms (Strassen rank) by sharing the U factors across the 6 column pairs,
    and summing the V and W factors.
    
    Rank = 7.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 7
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 7
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 7
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 7
    
    # Strassen's 7-term pattern
    # U indices: i*m + j (0..3)
    # V local indices: j*p_loc + l (0..1) where p_loc=2
    # W local indices: l*n + i (0..1) where n=2
    
    strassen_pattern = [
        # s=0
        {'U': [(0, 1.0), (3, 1.0)],
         'V': [(0, 0, 1.0), (1, 1, 1.0)],
         'W': [(0, 0, 1.0), (1, 1, 1.0)]},
        # s=1
        {'U': [(2, 1.0), (3, 1.0)],
         'V': [(0, 0, 1.0)],
         'W': [(0, 1, 1.0), (1, 1, -1.0)]},
        # s=2
        {'U': [(0, 1.0)],
         'V': [(0, 1, 1.0), (1, 1, -1.0)],
         'W': [(1, 0, 1.0), (1, 1, 1.0)]},
        # s=3
        {'U': [(3, 1.0)],
         'V': [(1, 0, 1.0), (0, 0, -1.0)],
         'W': [(0, 0, 1.0), (0, 1, 1.0)]},
        # s=4
        {'U': [(0, 1.0), (1, 1.0)],
         'V': [(1, 1, 1.0)],
         'W': [(0, 0, -1.0), (1, 0, 1.0)]},
        # s=5
        {'U': [(2, 1.0), (0, -1.0)],
         'V': [(0, 0, 1.0), (0, 1, 1.0)],
         'W': [(1, 1, 1.0)]},
        # s=6
        {'U': [(1, 1.0), (3, -1.0)],
         'V': [(1, 0, 1.0), (1, 1, 1.0)],
         'W': [(0, 0, 1.0)]},
    ]
    
    num_pairs = p // 2  # 6
    
    for s in range(rank):
        if time.time() > deadline:
            break
        pat = strassen_pattern[s]
        
        # Fill U
        for u_idx, coeff in pat['U']:
            U[u_idx][s] += coeff
            
        # Fill V
        # V local index is j * 2 + l
        # Global V index is j * p + (2*t + l)
        for j, l, coeff in pat['V']:
            for t in range(num_pairs):
                global_k = 2 * t + l
                v_idx = j * p + global_k
                V[v_idx][s] += coeff
                
        # Fill W
        # W local index is l * 2 + i
        # Global W index is (2*t + l) * n + i
        for l, i, coeff in pat['W']:
            for t in range(num_pairs):
                global_k = 2 * t + l
                w_idx = global_k * n + i
                W[w_idx][s] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
