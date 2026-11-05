"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define splits
    # n=3 -> [1, 2]
    # m=5 -> [1, 2, 2]
    # p=6 -> [2, 2, 2]
    n_splits = [1, 2]
    m_splits = [1, 2, 2]
    p_splits = [2, 2, 2]
    
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
        
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
        
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Calculate total rank
    total_rank = 0
    for i in range(len(n_splits)):
        for k in range(len(m_splits)):
            for j in range(len(p_splits)):
                ni = n_splits[i]
                mk = m_splits[k]
                pj = p_splits[j]
                if ni == 2 and mk == 2 and pj == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mk * pj
                    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(len(n_splits)):
        for k in range(len(m_splits)):
            for j in range(len(p_splits)):
                ni = n_splits[i]
                mk = m_splits[k]
                pj = p_splits[j]
                
                n_off = n_offsets[i]
                m_off = m_offsets[k]
                p_off = p_offsets[j]
                
                if ni == 2 and mk == 2 and pj == 2:
                    # Strassen 2x2x2
                    # M1
                    U[(n_off + 0) * m + (m_off + 0)][col] = 1
                    U[(n_off + 1) * m + (m_off + 1)][col] = 1
                    V[(m_off + 0) * p + (p_off + 0)][col] = 1
                    V[(m_off + 1) * p + (p_off + 1)][col] = 1
                    W[(p_off + 0) * n + (n_off + 0)][col] = 1
                    W[(p_off + 1) * n + (n_off + 1)][col] = 1
                    col += 1
                    
                    # M2
                    U[(n_off + 1) * m + (m_off + 0)][col] = 1
                    U[(n_off + 1) * m + (m_off + 1)][col] = 1
                    V[(m_off + 0) * p + (p_off + 0)][col] = 1
                    W[(p_off + 0) * n + (n_off + 1)][col] = 1
                    W[(p_off + 1) * n + (n_off + 1)][col] = -1
                    col += 1
                    
                    # M3
                    U[(n_off + 0) * m + (m_off + 0)][col] = 1
                    V[(m_off + 1) * p + (p_off + 0)][col] = 1
                    V[(m_off + 1) * p + (p_off + 1)][col] = -1
                    W[(p_off + 1) * n + (n_off + 0)][col] = 1
                    W[(p_off + 1) * n + (n_off + 1)][col] = 1
                    col += 1
                    
                    # M4
                    U[(n_off + 1) * m + (m_off + 1)][col] = 1
                    V[(m_off + 0) * p + (p_off + 1)][col] = 1
                    V[(m_off + 0) * p + (p_off + 0)][col] = -1
                    W[(p_off + 0) * n + (n_off + 0)][col] = 1
                    W[(p_off + 0) * n + (n_off + 1)][col] = 1
                    col += 1
                    
                    # M5
                    U[(n_off + 0) * m + (m_off + 0)][col] = 1
                    U[(n_off + 1) * m + (m_off + 0)][col] = 1
                    V[(m_off + 1) * p + (p_off + 1)][col] = 1
                    W[(p_off + 0) * n + (n_off + 0)][col] = -1
                    W[(p_off + 1) * n + (n_off + 0)][col] = 1
                    col += 1
                    
                    # M6
                    U[(n_off + 1) * m + (m_off + 1)][col] = 1
                    U[(n_off + 0) * m + (m_off + 0)][col] = -1
                    V[(m_off + 0) * p + (p_off + 0)][col] = 1
                    V[(m_off + 1) * p + (p_off + 0)][col] = 1
                    W[(p_off + 1) * n + (n_off + 1)][col] = 1
                    col += 1
                    
                    # M7
                    U[(n_off + 0) * m + (m_off + 1)][col] = 1
                    U[(n_off + 1) * m + (m_off + 1)][col] = -1
                    V[(m_off + 1) * p + (p_off + 1)][col] = 1
                    V[(m_off + 0) * p + (p_off + 1)][col] = 1
                    W[(p_off + 0) * n + (n_off + 0)][col] = 1
                    W[(p_off + 1) * n + (n_off + 1)][col] = 1
                    col += 1
                else:
                    # Schoolbook
                    for a in range(ni):
                        for b in range(mk):
                            for c in range(pj):
                                U[(n_off + a) * m + (m_off + b)][col] = 1
                                V[(m_off + b) * p + (p_off + c)][col] = 1
                                W[(p_off + c) * n + (n_off + a)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
