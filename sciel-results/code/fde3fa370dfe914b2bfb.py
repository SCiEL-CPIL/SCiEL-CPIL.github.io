import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_2x2_factors():
        terms = []
        # M1
        terms.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        ))
        # M2
        terms.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,0):1, (1,1):-1}
        ))
        # M3
        terms.append((
            {(0,0):1},
            {(1,0):1, (1,1):-1},
            {(0,1):1, (1,1):1}
        ))
        # M4
        terms.append((
            {(1,1):1},
            {(0,1):1, (0,0):-1},
            {(0,0):1, (1,0):1}
        ))
        # M5
        terms.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,0):-1, (0,1):1}
        ))
        # M6
        terms.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ))
        # M7
        terms.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1}
        ))
        return terms

    s2_terms = get_strassen_2x2_factors()
    
    col = 0
    for t_idx, (u_t, v_t, w_t) in enumerate(s2_terms):
        for s_idx, (u_s, v_s, w_s) in enumerate(s2_terms):
            U_dict = {}
            for (lr, lc), coeff_s in u_s.items():
                coeff_t = u_t.get((lr, lc), 0)
                if coeff_t != 0 and coeff_s != 0:
                    for i in range(2*lr, 2*lr+2):
                        for j in range(2*lc, 2*lc+2):
                            U_dict[(i, j)] = U_dict.get((i, j), 0) + coeff_s * coeff_t

            V_dict = {}
            for (lr, lc), coeff_s in v_s.items():
                coeff_t = v_t.get((lr, lc), 0)
                if coeff_t != 0 and coeff_s != 0:
                    for j in range(2*lr, 2*lr+2):
                        for k in range(2*lc, 2*lc+2):
                            V_dict[(j, k)] = V_dict.get((j, k), 0) + coeff_s * coeff_t

            W_dict = {}
            for (lr, lc), coeff_s in w_s.items():
                coeff_t = w_t.get((lr, lc), 0)
                if coeff_t != 0 and coeff_s != 0:
                    for k in range(2*lr, 2*lr+2):
                        for i in range(2*lc, 2*lc+2):
                            W_dict[(k, i)] = W_dict.get((k, i), 0) + coeff_s * coeff_t

            for (i, j), val in U_dict.items():
                U[i*5 + j][col] += val
            for (j, k), val in V_dict.items():
                V[j*5 + k][col] += val
            for (k, i), val in W_dict.items():
                W[k*4 + i][col] += val
                
            col += 1

    # Outer product for j=4
    for i in range(4):
        U[i*5 + 4][col] = 1
        for k in range(5):
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
    col += 1

    # Dot products for j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
