DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    g1 = [0, 1, 2]  # size 3
    g2 = [3, 4]     # size 2
    
    terms = []
    
    def make_vec(dimension, indices_values):
        vec = [0] * dimension
        for idx, val in indices_values:
            vec[idx] = val
        return vec
    
    def add_term(U_vec, V_vec, W_vec):
        terms.append((U_vec, V_vec, W_vec))
    
    # Block 1: 3x3 core (indices 0,1,2) - schoolbook
    for i in g1:
        for j in g1:
            for k in g1:
                U_vec = make_vec(25, [(i*5+j, 1)])
                V_vec = make_vec(25, [(j*5+k, 1)])
                W_vec = make_vec(25, [(k*5+i, 1)])
                add_term(U_vec, V_vec, W_vec)
    
    # Block 2: 2x2 core (indices 3,4) - Strassen
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # M7
    ]
    
    for U_local, V_local, W_local in strassen_2x2:
        U_vec = make_vec(25, [(g2[i]*5+g2[j], coeff) for (i,j), coeff in U_local.items()])
        V_vec = make_vec(25, [(g2[j]*5+g2[k], coeff) for (j,k), coeff in V_local.items()])
        W_vec = make_vec(25, [(g2[k]*5+g2[i], coeff) for (i,k), coeff in W_local.items()])
        add_term(U_vec, V_vec, W_vec)
    
    # Cross-block 1: A_12 (3x2) * B_21 (2x3) -> C_11 (3x3)
    for i in g1:
        for j in g2:
            for k in g1:
                U_vec = make_vec(25, [(i*5+j, 1)])
                V_vec = make_vec(25, [(j*5+k, 1)])
                W_vec = make_vec(25, [(k*5+i, 1)])
                add_term(U_vec, V_vec, W_vec)
    
    # Cross-block 2: A_12 (3x2) * B_22 (2x2) -> C_12 (3x2)
    for i in g1:
        for j in g2:
            for k in g2:
                U_vec = make_vec(25, [(i*5+j, 1)])
                V_vec = make_vec(25, [(j*5+k, 1)])
                W_vec = make_vec(25, [(k*5+i, 1)])
                add_term(U_vec, V_vec, W_vec)
    
    # Cross-block 3: A_21 (2x3) * B_11 (3x3) -> C_21 (2x3)
    for i in g2:
        for j in g1:
            for k in g1:
                U_vec = make_vec(25, [(i*5+j, 1)])
                V_vec = make_vec(25, [(j*5+k, 1)])
                W_vec = make_vec(25, [(k*5+i, 1)])
                add_term(U_vec, V_vec, W_vec)
    
    # Cross-block 4: A_21 (2x3) * B_12 (3x2) -> C_22 (2x2)
    for i in g2:
        for j in g1:
            for k in g2:
                U_vec = make_vec(25, [(i*5+j, 1)])
                V_vec = make_vec(25, [(j*5+k, 1)])
                W_vec = make_vec(25, [(k*5+i, 1)])
                add_term(U_vec, V_vec, W_vec)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col, (U_vec, V_vec, W_vec) in enumerate(terms):
        for row in range(25):
            U[row][col] = U_vec[row]
            V[row][col] = V_vec[row]
            W[row][col] = W_vec[row]
    
    return {"U": U, "V": V, "W": W}
