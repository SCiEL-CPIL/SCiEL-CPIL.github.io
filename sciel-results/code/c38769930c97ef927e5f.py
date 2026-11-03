"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    # Calculate offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Initialize matrices
    # We don't know the rank yet, so we'll use a list of tuples
    # or just build the matrices dynamically.
    # Max rank estimate: 75
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    rank = 0
    
    # Strassen patterns for 2x2x2
    # Each pattern is (U_entries, V_entries, W_entries)
    # Entries are list of (local_i, local_j, coeff) etc.
    # U is 2x2, V is 2x2, W is 2x2 (indexed k, i)
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,0,1), (1,1,1)
    # V: (0,0,1), (1,1,1)
    # W: (0,0,1), (1,1,1)
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00
        # U: (1,0,1), (1,1,1)
        # V: (0,0,1)
        # W: (0,1,1), (1,1,-1)
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3: A00(B01-B11)
        # U: (0,0,1)
        # V: (0,1,1), (1,1,-1)
        # W: (1,0,1), (1,1,1)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4: A11(B10-B00)
        # U: (1,1,1)
        # V: (1,0,1), (0,0,-1)
        # W: (0,0,1), (0,1,1)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5: (A00+A01)B11
        # U: (0,0,1), (0,1,1)
        # V: (1,1,1)
        # W: (0,0,-1), (1,0,1)
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6: (A10-A00)(B00+B01)
        # U: (1,0,1), (0,0,-1)
        # V: (0,0,1), (0,1,1)
        # W: (1,1,1)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11)
        # U: (0,1,1), (1,1,-1)
        # V: (1,0,1), (1,1,1)
        # W: (0,0,1)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                n_b = n_splits[i_b]
                m_b = m_splits[j_b]
                p_b = p_splits[k_b]
                
                i_off = n_offsets[i_b]
                j_off = m_offsets[j_b]
                k_off = p_offsets[k_b]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    for pattern in strassen_patterns:
                        col = rank
                        # U
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] = val
                        # V
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] = val
                        # W
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] = val
                        rank += 1
                else:
                    # Schoolbook
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
                                col = rank
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                rank += 1
    
    # Trim matrices to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
