def construct():
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen block coefficients
    # U: (i, j)
    # V: (j, k)
    # W: (k, i)
    
    # Term 0: M1 = A00(B00 - B11)
    # U: (0,0)=1
    # V: (0,0)=1, (1,1)=-1
    # W: (0,0)=1, (0,1)=1  (C00, C10)
    
    # I will define a helper to add a term
    def add_term(r, u_dict, v_dict, w_dict):
        for (i, j), val in u_dict.items():
            U[i*m + j][r] += val
        for (j, k), val in v_dict.items():
            V[j*p + k][r] += val
        for (k, i), val in w_dict.items():
            W[k*n + i][r] += val

    # Strassen Terms
    # M1
    add_term(0, {(0,0): 1}, {(0,0): 1, (1,1): -1}, {(0,0): 1, (0,1): 1})
    # M2
    add_term(1, {(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1})
    # M3
    add_term(2, {(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1, (0,1): 1})
    # M4
    add_term(3, {(1,1): 1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1})
    # M5
    add_term(4, {(0,0): 1, (1,0): 1}, {(1,0): 1}, {(1,0): 1, (1,1): 1})
    # M6
    add_term(5, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (1,1): 1})
    # M7
    add_term(6, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): -1})
    
    # Wait, the indices in W are (k, i).
    # C_ik.
    # M1 contributes to C_00 and C_10.
    # C_00: i=0, k=0. W index 0*n + 0 = 0.
    # C_10: i=1, k=0. W index 0*n + 1 = 1.
    # So {(0,0): 1, (0,1): 1} is correct for W.
    
    # I need to repeat this for blocks 1-4.
    # The U part is the same.
    # The V and W parts shift k by 2*b.
    
    for b in range(5):
        base_r = b * 7
        # M1
        add_term(base_r + 0, {(0,0): 1}, {(0, 2*b): 1, (1, 2*b+1): -1}, {(2*b, 0): 1, (2*b, 1): 1})
        # M2
        add_term(base_r + 1, {(0,0): 1, (0,1): 1}, {(1, 2*b+1): 1}, {(2*b+1, 0): 1, (2*b+1, 1): 1})
        # M3
        add_term(base_r + 2, {(1,0): 1, (1,1): 1}, {(0, 2*b): 1}, {(2*b, 0): 1, (2*b, 1): 1})
        # M4
        add_term(base_r + 3, {(1,1): 1}, {(1, 2*b+1): 1, (0, 2*b): -1}, {(2*b, 0): 1, (2*b+1, 1): 1})
        # M5
        add_term(base_r + 4, {(0,0): 1, (1,0): 1}, {(1, 2*b): 1}, {(2*b, 1): 1, (2*b+1, 1): 1})
        # M6
        add_term(base_r + 5, {(0,1): 1, (1,1): -1}, {(0, 2*b+1): 1, (1, 2*b+1): 1}, {(2*b, 0): -1, (2*b+1, 1): 1})
        # M7
        add_term(base_r + 6, {(0,0): 1, (1,0): -1}, {(0, 2*b): 1, (1, 2*b): 1}, {(2*b, 0): 1, (2*b, 1): -1})
        
    # Last block (col 10)
    # Rank 4
    # T0: A00 B00 -> C00
    add_term(35, {(0,0): 1}, {(0, 10): 1}, {(10, 0): 1})
    # T1: A01 B10 -> C00
    add_term(36, {(0,1): 1}, {(1, 10): 1}, {(10, 0): 1})
    # T2: A10 B00 -> C10
    add_term(37, {(1,0): 1}, {(0, 10): 1}, {(10, 1): 1})
    # T3: A11 B10 -> C10
    add_term(38, {(1,1): 1}, {(1, 10): 1}, {(10, 1): 1})
    
    return {"U": U, "V": V, "W": W}
