"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    terms = []
    
    def add_term(u_dict, v_dict, w_dict, coeff=1):
        terms.append((u_dict, v_dict, w_dict, coeff))
    
    def add_strassen_2x2x2(X_ro, X_co, Y_ro, Y_co, C_ro, C_co, sign=1):
        u1 = {(X_ro, X_co): sign, (X_ro+1, X_co+1): sign}
        v1 = {(Y_ro, Y_co): sign, (Y_ro+1, Y_co+1): sign}
        w1 = {(C_co, C_ro): sign, (C_co+1, C_ro+1): sign}
        add_term(u1, v1, w1, 1)
        
        u2 = {(X_ro+1, X_co): sign, (X_ro+1, X_co+1): sign}
        v2 = {(Y_ro, Y_co): sign}
        w2 = {(C_co, C_ro+1): sign, (C_co+1, C_ro+1): sign}
        add_term(u2, v2, w2, 1)
        
        u3 = {(X_ro, X_co): sign}
        v3 = {(Y_ro, Y_co+1): sign, (Y_ro+1, Y_co+1): -sign}
        w3 = {(C_co+1, C_ro): sign, (C_co+1, C_ro+1): sign}
        add_term(u3, v3, w3, 1)
        
        u4 = {(X_ro+1, X_co+1): sign}
        v4 = {(Y_ro+1, Y_co): sign, (Y_ro, Y_co): -sign}
        w4 = {(C_co, C_ro+1): sign, (C_co+1, C_ro+1): sign}
        add_term(u4, v4, w4, 1)
        
        u5 = {(X_ro, X_co): sign, (X_ro, X_co+1): sign}
        v5 = {(Y_ro+1, Y_co+1): sign}
        w5 = {(C_co, C_ro): -sign, (C_co+1, C_ro): sign}
        add_term(u5, v5, w5, 1)
        
        u6 = {(X_ro+1, X_co): sign, (X_ro, X_co): -sign}
        v6 = {(Y_ro, Y_co): sign, (Y_ro, Y_co+1): sign}
        w6 = {(C_co+1, C_ro+1): sign}
        add_term(u6, v6, w6, 1)
        
        u7 = {(X_ro, X_co+1): sign, (X_ro+1, X_co+1): -sign}
        v7 = {(Y_ro+1, Y_co): sign, (Y_ro+1, Y_co+1): sign}
        w7 = {(C_co, C_ro): sign, (C_co+1, C_ro+1): sign}
        add_term(u7, v7, w7, 1)
    
    # Part 1: 4x4x4
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0, 1)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0, 1)
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2, 1)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2, 1)
    add_strassen_2x2x2(2, 0, 0, 0, 2, 0, 1)
    add_strassen_2x2x2(2, 2, 2, 0, 2, 0, 1)
    add_strassen_2x2x2(2, 0, 0, 2, 2, 2, 1)
    add_strassen_2x2x2(2, 2, 2, 2, 2, 2, 1)
    
    # Part 2: 4x4x3 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                add_term(u, v, w, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict, coeff) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] += val * coeff
        for (j, k), val in v_dict.items():
            V[j * p + k][c] += val * coeff
        for (k, i), val in w_dict.items():
            W[k * n + i][c] += val * coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

