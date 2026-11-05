import time

def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 5, 6
    
    # Strassen 2x2x2 terms (local indices 0,1)
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Group indices
    i_groups = [(0,1), (2,3)]
    j_groups = [(0,1), (2,3), (4,)]
    k_groups = [(0,1), (2,3), (4,5)]
    
    all_terms = []
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                i0, i1 = ig[0], ig[1]
                k0, k1 = kg[0], kg[1]
                
                if len(jg) == 2:
                    j0, j1 = jg[0], jg[1]
                    for (U_loc, V_loc, W_loc) in strassen_terms:
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for (li, lj), c in U_loc.items():
                            U_dict[(i0+li, j0+lj)] = c
                        for (lj, lk), c in V_loc.items():
                            V_dict[(j0+lj, k0+lk)] = c
                        for (lk, li), c in W_loc.items():
                            W_dict[(k0+lk, i0+li)] = c
                        all_terms.append((U_dict, V_dict, W_dict))
                else:
                    j0 = jg[0]
                    for i in [i0, i1]:
                        for k in [k0, k1]:
                            all_terms.append(
                                ({(i, j0): 1}, {(j0, k): 1}, {(k, i): 1})
                            )
    
    rank = len(all_terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), c in U_dict.items():
            U[i*m + j][col] = c
        for (j, k), c in V_dict.items():
            V[j*p + k][col] = c
        for (k, i), c in W_dict.items():
            W[k*n + i][col] = c
    
    return {"U": U, "V": V, "W": W}
