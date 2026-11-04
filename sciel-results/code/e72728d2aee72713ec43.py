import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    deadline = time.time() + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    strassen = try_strassen_blocks(n, m, p)
    if strassen is not None:
        sr = len(strassen['U'][0])
        if sr < best_rank:
            best_result = strassen
            best_rank = sr
    
    return best_result

def try_strassen_blocks(n, m, p):
    if n != 3 or m != 4 or p != 4:
        return None
    
    U_list = []
    V_list = []
    W_list = []
    col = 0
    
    def add_strassen_2x2(i_base, j_base, k_base):
        nonlocal col
        a11 = (i_base, j_base)
        a12 = (i_base, j_base+1)
        a21 = (i_base+1, j_base)
        a22 = (i_base+1, j_base+1)
        b11 = (j_base, k_base)
        b12 = (j_base, k_base+1)
        b21 = (j_base+1, k_base)
        b22 = (j_base+1, k_base+1)
        c11 = (i_base, k_base)
        c12 = (i_base, k_base+1)
        c21 = (i_base+1, k_base)
        c22 = (i_base+1, k_base+1)
        
        U_list.append(([a11, a22], [1, 1]))
        V_list.append(([b11, b22], [1, 1]))
        W_list.append(([c11, c22], [1, 1]))
        col += 1
        
        U_list.append(([a21, a22], [1, 1]))
        V_list.append(([b11], [1]))
        W_list.append(([c21, c22], [1, -1]))
        col += 1
        
        U_list.append(([a11], [1]))
        V_list.append(([b21, b22], [1, -1]))
        W_list.append(([c12, c22], [1, 1]))
        col += 1
        
        U_list.append(([a22], [1]))
        V_list.append(([b12, b11], [1, -1]))
        W_list.append(([c11, c21], [1, 1]))
        col += 1
        
        U_list.append(([a11, a12], [1, 1]))
        V_list.append(([b22], [1]))
        W_list.append(([c11, c12], [-1, 1]))
        col += 1
        
        U_list.append(([a21, a11], [1, -1]))
        V_list.append(([b11, b12], [1, 1]))
        W_list.append(([c22], [1]))
        col += 1
        
        U_list.append(([a12, a22], [1, -1]))
        V_list.append(([b21, b22], [1, 1]))
        W_list.append(([c11], [1]))
        col += 1
    
    add_strassen_2x2(0, 0, 0)
    add_strassen_2x2(0, 2, 0)
    add_strassen_2x2(0, 0, 2)
    add_strassen_2x2(0, 2, 2)
    
    for j in range(4):
        for k in range(4):
            U_list.append([[(2, j)]], [1])
            V_list.append([[(j, k)]], [1])
            W_list.append([[(k, 2)]], [1])
            col += 1
    
    total_rank = col
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for c in range(total_rank):
        for (idx, sign) in zip(U_list[c][0], U_list[c][1]):
            i, j = idx
            U[i*m + j, c] = sign
        for (idx, sign) in zip(V_list[c][0], V_list[c][1]):
            j, k = idx
            V[j*p + k, c] = sign
        for (idx, sign) in zip(W_list[c][0], W_list[c][1]):
            k, i = idx
            W[k*n + i, c] = sign
    
    if not verify_full(n, m, p, U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_full(n, m, p, U, V, W):
    tol = 1e-8
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True
