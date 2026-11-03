def construct():
    n, m, p = 4, 5, 6
    rank = 108
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Partitions
    n_starts = [0, 2]
    m_starts = [0, 2, 4]
    m_sizes = [2, 2, 1]
    p_starts = [0, 2, 4]
    
    col = 0
    for I in range(2):
        i0 = n_starts[I]
        for J in range(3):
            j0 = m_starts[J]
            js = m_sizes[J]
            for K in range(3):
                k0 = p_starts[K]
                
                if js == 2:
                    # Strassen 2x2x2
                    # 7 terms
                    for r in range(7):
                        # Get u, v, w coefficients for this term
                        # u is 2x2, v is 2x2, w is 2x2 (conceptually)
                        # We need to map local (i, j, k) to global
                        pass
                else:
                    # Schoolbook 2x1x2
                    # 4 terms
                    pass
                col += (7 if js == 2 else 4)
                
    return {"U": U, "V": V, "W": W}
