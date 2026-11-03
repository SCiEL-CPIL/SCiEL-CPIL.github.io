import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 

    def get_strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1
        
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] += -1
        
        U[0][2] += 1
        V[1][2] += 1; V[3][2] += -1
        W[1][2] += 1; W[3][2] += 1
        
        U[3][3] += 1
        V[2][3] += 1; V[0][3] += -1
        W[0][3] += 1; W[2][3] += 1
        
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] += -1; W[1][4] += 1
        
        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        U[1][6] += 1; U[3][6] += -1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1; W[3][6] += 1
        
        return U, V, W

    def get_schoolbook_2x2_2x1():
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W

    def get_decomp_2x2_2x3():
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        Us, Vs, Ws = get_strassen_2x2()
        
        for r in range(4):
            for c in range(7):
                U[r][c] += Us[r][c]
                
        v_map_s = [0, 1, 3, 4]
        for r_local in range(4):
            r_global = v_map_s[r_local]
            for c in range(7):
                V[r_global][c] += Vs[r_local][c]
                
        w_map_s = [0, 1, 3, 4]
        for r_local in range(4):
            r_global = w_map_s[r_local]
            for c in range(7):
                W[r_global][c] += Ws[r_local][c]
                
        Ub, Vb, Wb = get_schoolbook_2x2_2x1()
        
        for r in range(4):
            for c in range(4):
                U[r][c+7] += Ub[r][c]
                
        v_map_b = [2, 5]
        for r_local in range(2):
            r_global = v_map_b[r_local]
            for c in range(4):
                V[r_global][c+7] += Vb[r_local][c]
                
        w_map_b = [2, 5]
        for r_local in range(2):
            r_global = w_map_b[r_local]
            for c in range(4):
                W[r_global][c+7] += Wb[r_local][c]
                
        return U, V, W

    U_local, V_local, W_local = get_decomp_2x2_2x3()
    
    rank = 77
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    def add_factor(U_g, V_g, W_g, U_l, V_l, W_l, 
                   u_offsets, v_offsets, w_offsets, 
                   u_signs, v_signs, w_signs, 
                   col_start):
        for r in range(4):
            for c in range(11):
                val = U_l[r][c]
                if val != 0:
                    for i, off in enumerate(u_offsets):
                        U_g[off + r][col_start + c] += u_signs[i] * val
                        
        for r in range(6):
            for c in range(11):
                val = V_l[r][c]
                if val != 0:
                    for i, off in enumerate(v_offsets):
                        V_g[off + r][col_start + c] += v_signs[i] * val
                        
        for r in range(6):
            for c in range(11):
                val = W_l[r][c]
                if val != 0:
                    for i, off in enumerate(w_offsets):
                        W_g[off + r][col_start + c] += w_signs[i] * val

    add_factor(U, V, W, U_local, V_local, W_local,
               [0, 12], [0, 18], [0, 6, 12, 18],
               [1, 1], [1, 1], [1, 1, 1, 1], 0)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [8, 12], [0], [12, 18],
               [1, 1], [1], [1, -1], 11)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [0], [6, 18], [6, 18],
               [1], [1, -1], [1, 1], 22)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [12], [12, 0], [0, 12],
               [1], [1, -1], [1, 1], 33)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [0, 4], [18], [0, 6],
               [1, 1], [1], [-1, 1], 44)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [8, 0], [0, 6], [18],
               [1, -1], [1, 1], [1], 55)
               
    add_factor(U, V, W, U_local, V_local, W_local,
               [4, 12], [12, 18], [0, 18],
               [1, -1], [1, 1], [1, 1], 66)
               
    return {"U": U, "V": V, "W": W}
