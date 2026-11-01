import numpy as np

def construct():
    n, m, p = 2, 2, 14
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen matrices for 2x2x2
    # u, v, w are 2x2
    U_s = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[1,0],[1,0]],
        [[0,1],[0,1]]
    ]
    V_s = [
        [[1,0],[0,1]],
        [[1,1],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,0],[-1,0]],
        [[0,1],[-1,0]]
    ]
    W_s = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[0,1],[0,0]],
        [[0,0],[0,1]],
        [[0,0],[0,1]],
        [[0,1],[0,0]],
        [[1,0],[0,0]]
    ]
    
    col_idx = 0
    for q in range(7):
        k1 = 2*q
        k2 = 2*q + 1
        for t in range(7):
            u_t = U_s[t]
            v_t = V_s[t]
            w_t = W_s[t]
            
            # Fill U
            # U index i*m + j
            for i in range(2):
                for j in range(2):
                    U[i*2 + j, col_idx] = u_t[i][j]
                    
            # Fill V
            # V index j*p + k
            for j in range(2):
                V[j*14 + k1, col_idx] = v_t[j][0]
                V[j*14 + k2, col_idx] = v_t[j][1]
                
            # Fill W
            # W index k*n + i
            for i in range(2):
                W[k1*2 + i, col_idx] = w_t[0][i]
                W[k2*2 + i, col_idx] = w_t[1][i]
                
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
