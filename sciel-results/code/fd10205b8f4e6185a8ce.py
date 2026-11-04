def construct():
    n, m, p = 4, 4, 5
    rank = 28 + 16  # Strassen for 4x4x4 + schoolbook for 4x4x1
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen coefficients
    coeff_A = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    coeff_B = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    
    # 4x4x4 part: Strassen, rank 28
    for s in range(7):
        for r in range(2):
            for b in range(2):
                c_idx = s * 4 + r * 2 + b
                for i in range(4):
                    for j in range(4):
                        ib, io = i // 2, i % 2
                        jb, jo = j // 2, j % 2
                        if io == r and jo == b:
                            U[i*m + j][c_idx] = coeff_A[s].get((ib, jb), 0)
                for j in range(4):
                    for k in range(4):
                        jb, jo = j // 2, j % 2
                        kb, ko = k // 2, k % 2
                        if jo == b and ko == r:
                            V[j*m + k][c_idx] = coeff_B[s].get((jb, kb), 0)
                for k in range(4):
                    for i in range(4):
                        kb, ko = k // 2, k % 2
                        ib, io = i // 2, i % 2
                        if ko == r and io == r:
                            W[k*n + i][c_idx] = 1
    
    # 4x4x1 part: schoolbook, rank 16
    for i in range(4):
        for j in range(4):
            c_idx = 28 + i * 4 + j
            U[i*m + j][c_idx] = 1
            V[j*m + 4][c_idx] = 1
            W[4*n + i][c_idx] = 1
    
    return {"U": U, "V": V, "W": W}
