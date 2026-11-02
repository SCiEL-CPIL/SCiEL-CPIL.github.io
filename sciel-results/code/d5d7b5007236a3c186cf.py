def construct():
    n, m, p = 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 23 for 3x3x3 (Laderman)
    rank_laderman = 23
    # Rank 9 for 3x3x1 (Schoolbook)
    rank_vec = 9
    
    total_rank = rank_laderman + rank_vec
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    # Laderman factors for 3x3x3
    # U_L, V_L, W_L are 9x23
    # I will hardcode these.
    # ...
