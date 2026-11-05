import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # 2x2x2 Strassen factors
    s_u = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ]
    s_v = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    s_w = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    # 4x4x4 part: 7 outer terms, each with 7 inner terms
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        for inner_k in range(7):
            col_idx = t_idx * 7 + inner_k
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            for (a_br, a_bc), a_c in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        if u_vec[local_idx] != 0:
                            gi = 2*a_br + lr
                            gj = 2*a_bc + lc
                            U[gi*m+gj][col_idx] += a_c * u_vec[local_idx]
            
            for (b_br, b_bc), b_c in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        if v_vec[local_idx] != 0:
                            gj = 2*b_br + lr
                            gk = 2*b_bc + lc
                            V[gj*p+gk][col_idx] += b_c * v_vec[local_idx]
            
            for (c_br, c_bc), c_c in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        if w_vec[local_idx] != 0:
                            gk = 2*c_bc + lc
                            gi = 2*c_br + lr
                            W[gk*n+gi][col_idx] += c_c * w_vec[local_idx]
    
    col = 49
    # 4x4x1 part
    for i in range(4):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    # 4x1x4 part
    for i in range(4):
        for k in range(4):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # 4x1x1 part
    for i in range(4):
        U[i*m+4][col] = 1
        V[4*p+4][col] = 1
        W[4*n+i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
