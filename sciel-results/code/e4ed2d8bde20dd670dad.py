def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,0,1,1]),
        ([0,0,0,1], [0,1,-1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    col = 0
    subproblems = [
        (0, 0, 0, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 0),
        (0, 2, 2, 2),
    ]
    
    for (ar, ac, br, bc) in subproblems:
        for (u_loc, v_loc, w_loc) in strassen_terms:
            for u_idx in range(4):
                i = u_idx // 2
                j = u_idx % 2
                if u_loc[u_idx] != 0:
                    U[(ar+i)*4 + (ac+j)][col] = u_loc[u_idx]
            for v_idx in range(4):
                j = v_idx // 2
                k = v_idx % 2
                if v_loc[v_idx] != 0:
                    V[(br+j)*4 + (bc+k)][col] = v_loc[v_idx]
            for w_idx in range(4):
                k = w_idx // 2
                i = w_idx % 2
                if w_loc[w_idx] != 0:
                    W[(bc+k)*3 + (ar+i)][col] = w_loc[w_idx]
            col += 1
    
    for i in range(4):
        for j in range(4):
            U[2*4 + i][col] = 1
            V[i*4 + j][col] = 1
            W[j*3 + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
