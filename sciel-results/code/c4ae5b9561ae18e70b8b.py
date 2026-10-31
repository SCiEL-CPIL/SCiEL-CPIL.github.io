def construct():
    n, m, p = DIMENSIONS
    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,0], [0,0,1,1], [1,0,0,0]),
        ([0,1,0,0], [1,0,0,0], [0,0,1,-1]),
        ([1,0,1,0], [0,0,0,1], [0,1,-1,0]),
        ([1,0,0,-1], [1,1,0,0], [0,0,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,0,0,1], [0,1,0,-1], [0,1,0,1]),
    ]
    def a_index(i_o, i_i, k_o, k_i):
        return i_o*8 + i_i*4 + k_o*2 + k_i
    def b_index(k_o, k_i, j_o, j_i):
        return k_o*8 + k_i*4 + j_o*2 + j_i
    def w_index(j_o, j_i, i_o, i_i):
        return j_o*8 + j_i*4 + i_o*2 + i_i
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for (u_o, v_o, w_o) in strassen:
        for (u_i, v_i, w_i) in strassen:
            for i_o in range(2):
                for k_o in range(2):
                    for i_i in range(2):
                        for k_i in range(2):
                            a_val = v_o[i_o*2+k_o] * v_i[i_i*2+k_i]
                            if a_val != 0:
                                U[a_index(i_o,i_i,k_o,k_i)][col] = a_val
            for k_o in range(2):
                for j_o in range(2):
                    for k_i in range(2):
                        for j_i in range(2):
                            b_val = w_o[k_o*2+j_o] * w_i[k_i*2+j_i]
                            if b_val != 0:
                                V[b_index(k_o,k_i,j_o,j_i)][col] = b_val
            for j_o in range(2):
                for i_o in range(2):
                    for j_i in range(2):
                        for i_i in range(2):
                            w_val = u_o[i_o*2+j_o] * u_i[i_i*2+j_i]
                            if w_val != 0:
                                W[w_index(j_o,j_i,i_o,i_i)][col] = w_val
            col += 1
    return {"U": U, "V": V, "W": W}
