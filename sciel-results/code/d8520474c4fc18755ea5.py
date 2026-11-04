import numpy as np

def construct():
    n, m, p = 3, 4, 4
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, val):
        U_list.append([0.0] * (n*m))
        V_list.append([0.0] * (m*p))
        W_list.append([0.0] * (p*n))
        U_list[-1][u_idx] = val
        V_list[-1][v_idx] = val
        W_list[-1][w_idx] = val

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {'U': U, 'V': V, 'W': W}
