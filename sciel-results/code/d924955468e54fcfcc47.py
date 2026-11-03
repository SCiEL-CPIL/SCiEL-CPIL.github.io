import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add 2x2x2 Strassen block
    def add_222(u_off, v_off, w_off, start_col):
        # u_off: [u00, u01, u10, u11]
        # v_off: [v00, v01, v10, v11]
        # w_off: [w00, w01, w10, w11]
        
        # M1
        U[u_off[0]][start_col] += 1
        U[u_off[3]][start_col] += 1
        V[v_off[0]][start_col] += 1
        V[v_off[3]][start_col] += 1
        W[w_off[0]][start_col] += 1
        W[w_off[3]][start_col] += 1
        start_col += 1
        
        # M2
        U[u_off[2]][start_col] += 1
        U[u_off[3]][start_col] += 1
        V[v_off[0]][start_col] += 1
        W[w_off[2]][start_col] += 1
        W[w_off[3]][start_col] -= 1
        start_col += 1
        
        # M3
        U[u_off[0]][start_col] += 1
        V[v_off[1]][start_col] += 1
        V[v_off[3]][start_col] -= 1
        W[w_off[1]][start_col] += 1
        W[w_off[3]][start_col] += 1
        start_col += 1
        
        # M4
        U[u_off[3]][start_col] += 1
        V[v_off[2]][start_col] += 1
        V[v_off[0]][start_col] -= 1
        W[w_off[0]][start_col] += 1
        W[w_off[2]][start_col] += 1
        start_col += 1
        
        # M5
        U[u_off[0]][start_col] += 1
        U[u_off[1]][start_col] += 1
        V[v_off[3]][start_col] += 1
        W[w_off[0]][start_col] -= 1
        W[w_off[1]][start_col] += 1
        start_col += 1
        
        # M6
        U[u_off[2]][start_col] += 1
        U[u_off[0]][start_col] -= 1
        V[v_off[0]][start_col] += 1
        V[v_off[1]][start_col] += 1
        W[w_off[3]][start_col] += 1
        start_col += 1
        
        # M7
        U[u_off[1]][start_col] += 1
        U[u_off[3]][start_col] -= 1
        V[v_off[2]][start_col] += 1
        V[v_off[3]][start_col] += 1
        W[w_off[0]][start_col] += 1
        start_col += 1
        
        return start_col

    # Helper to add 2x2x1 Schoolbook block
    def add_221(u_off, v_off, w_off, start_col):
        # u_off: [u00, u01, u10, u11]
        # v_off: [v0, v1]
        # w_off: [w0, w1]
        
        # x00 y0 -> w0
        U[u_off[0]][start_col] += 1
        V[v_off[0]][start_col] += 1
        W[w_off[0]][start_col] += 1
        start_col += 1
        
        # x01 y1 -> w0
        U[u_off[1]][start_col] += 1
        V[v_off[1]][start_col] += 1
        W[w_off[0]][start_col] += 1
        start_col += 1
        
        # x10 y0 -> w1
        U[u_off[2]][start_col] += 1
        V[v_off[0]][start_col] += 1
        W[w_off[1]][start_col] += 1
        start_col += 1
        
        # x11 y1 -> w1
        U[u_off[3]][start_col] += 1
        V[v_off[1]][start_col] += 1
        W[w_off[1]][start_col] += 1
        start_col += 1
        
        return start_col

    # Block 1: A1 B1
    u1 = [0, 1, 4, 5]
    
    # B1a (cols 0,1)
    v1a = [0, 1, 5, 6]
    w1a = [0, 1, 2, 3]
    col = add_222(u1, v1a, w1a, col)
    
    # B1b (cols 2,3)
    v1b = [2, 3, 7, 8]
    w1b = [4, 5, 6, 7]
    col = add_222(u1, v1b, w1b, col)
    
    # B1c (col 4)
    v1c = [4, 9]
    w1c = [8, 9]
    col = add_221(u1, v1c, w1c, col)
    
    # Block 2: A2 B2
    u2 = [2, 3, 6, 7]
    
    # B2a (cols 0,1)
    v2a = [10, 11, 15, 16]
    w2a = [0, 1, 2, 3]
    col = add_222(u2, v2a, w2a, col)
    
    # B2b (cols 2,3)
    v2b = [12, 13, 17, 18]
    w2b = [4, 5, 6, 7]
    col = add_222(u2, v2b, w2b, col)
    
    # B2c (col 4)
    v2c = [14, 19]
    w2c = [8, 9]
    col = add_221(u2, v2c, w2c, col)
    
    return {"U": U, "V": V, "W": W}
