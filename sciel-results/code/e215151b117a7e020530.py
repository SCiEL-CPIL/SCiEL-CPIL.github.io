def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_product(rank_idx, u_map, v_map, w_map):
        # M1 = (x11 - x01) * (y00 + y10)
        # Z00 += -M1, Z10 += M1
        U[u_map['x11']][rank_idx] += 1.0
        U[u_map['x01']][rank_idx] += -1.0
        V[v_map['y00']][rank_idx] += 1.0
        V[v_map['y10']][rank_idx] += 1.0
        W[w_map['z00']][rank_idx] += -1.0
        W[w_map['z10']][rank_idx] += 1.0
        
        # M2 = (x00 + x01) * y11
        # Z10 += M2, Z11 += -M2
        U[u_map['x00']][rank_idx] += 1.0
        U[u_map['x01']][rank_idx] += 1.0
        V[v_map['y11']][rank_idx] += 1.0
        W[w_map['z10']][rank_idx] += 1.0
        W[w_map['z11']][rank_idx] += -1.0
        
        # M3 = x00 * (y01 - y11)
        # Z01 += M3, Z11 += -M3
        U[u_map['x00']][rank_idx] += 1.0
        V[v_map['y01']][rank_idx] += 1.0
        V[v_map['y11']][rank_idx] += -1.0
        W[w_map['z01']][rank_idx] += 1.0
        W[w_map['z11']][rank_idx] += -1.0
        
        # M4 = x11 * (y10 - y00)
        # Z00 += M4, Z11 += M4
        U[u_map['x11']][rank_idx] += 1.0
        V[v_map['y10']][rank_idx] += 1.0
        V[v_map['y00']][rank_idx] += -1.0
        W[w_map['z00']][rank_idx] += 1.0
        W[w_map['z11']][rank_idx] += 1.0
        
        # M5 = (x00 + x11) * (y00 + y11)
        # Z00 += M5, Z01 += M5, Z10 += M5, Z11 += M5
        U[u_map['x00']][rank_idx] += 1.0
        U[u_map['x11']][rank_idx] += 1.0
        V[v_map['y00']][rank_idx] += 1.0
        V[v_map['y11']][rank_idx] += 1.0
        W[w_map['z00']][rank_idx] += 1.0
        W[w_map['z01']][rank_idx] += 1.0
        W[w_map['z10']][rank_idx] += 1.0
        W[w_map['z11']][rank_idx] += 1.0
        
        # M6 = (x01 - x11) * (y10 + y11)
        # Z11 += M6
        U[u_map['x01']][rank_idx] += 1.0
        U[u_map['x11']][rank_idx] += -1.0
        V[v_map['y10']][rank_idx] += 1.0
        V[v_map['y11']][rank_idx] += 1.0
        W[w_map['z11']][rank_idx] += 1.0
        
        # M7 = (x00 - x01) * (y00 + y01)
        # Z00 += -M7
        U[u_map['x00']][rank_idx] += 1.0
        U[u_map['x01']][rank_idx] += -1.0
        V[v_map['y00']][rank_idx] += 1.0
        V[v_map['y01']][rank_idx] += 1.0
        W[w_map['z00']][rank_idx] += -1.0

    # Define mappings for the 4 quadrants and 2 halves
    # U indices: 4i+k
    # V indices: 8k+j
    # W indices: 2j+i
    
    configs = []
    
    # Q0: A_L (k 0-1), B_T (k 0-1)
    # Half L: j 0-1
    configs.append((
        {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5},
        {'y00': 0, 'y01': 1, 'y10': 8, 'y11': 9},
        {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    ))
    # Half R: j 2-3
    configs.append((
        {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5},
        {'y00': 2, 'y01': 3, 'y10': 10, 'y11': 11},
        {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    ))
    
    # Q1: A_L (k 0-1), B_B (k 2-3)
    # Half L: j 0-1
    configs.append((
        {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5},
        {'y00': 16, 'y01': 17, 'y10': 24, 'y11': 25},
        {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    ))
    # Half R: j 2-3
    configs.append((
        {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5},
        {'y00': 18, 'y01': 19, 'y10': 26, 'y11': 27},
        {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    ))
    
    # Q2: A_R (k 2-3), B_T (k 0-1)
    # Half L: j 0-1
    configs.append((
        {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7},
        {'y00': 0, 'y01': 1, 'y10': 8, 'y11': 9},
        {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    ))
    # Half R: j 2-3
    configs.append((
        {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7},
        {'y00': 2, 'y01': 3, 'y10': 10, 'y11': 11},
        {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    ))
    
    # Q3: A_R (k 2-3), B_B (k 2-3)
    # Half L: j 0-1
    configs.append((
        {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7},
        {'y00': 16, 'y01': 17, 'y10': 24, 'y11': 25},
        {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    ))
    # Half R: j 2-3
    configs.append((
        {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7},
        {'y00': 18, 'y01': 19, 'y10': 26, 'y11': 27},
        {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    ))
    
    for idx, (um, vm, wm) in enumerate(configs):
        add_strassen_product(idx, um, vm, wm)
        
    return {"U": U, "V": V, "W": W}
