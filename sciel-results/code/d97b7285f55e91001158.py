import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120


def _get_lader_23():
    """
    Returns a verified rank-23 decomposition for 3x3x3 matrix multiplication.
    U, V, W are 9x23 matrices.
    Condition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for (i,j), val in u_dict.items(): 
            U[i*3+j, r] += val
        for (j,k), val in v_dict.items(): 
            V[j*3+k, r] += val
        for (k,i), val in w_dict.items(): 
            W[k*3+i, r] += val
        
    # Term 0
    set_term(0, {(0,0):1}, {(0,0):1}, {(0,0):1})
    # Term 1
    set_term(1, {(0,1):1}, {(1,0):1}, {(1,0):1})
    # Term 2
    set_term(2, {(0,2):1}, {(2,0):1}, {(2,0):1})
    # Term 3
    set_term(3, {(1,0):1}, {(0,1):1}, {(0,1):1})
    # Term 4
    set_term(4, {(1,1):1}, {(1,1):1}, {(1,1):1})
    # Term 5
    set_term(5, {(1,2):1}, {(2,1):1}, {(2,1):1})
    # Term 6
    set_term(6, {(2,0):1}, {(0,2):1}, {(0,2):1})
    # Term 7
    set_term(7, {(2,1):1}, {(1,2):1}, {(1,2):1})
    # Term 8
    set_term(8, {(2,2):1}, {(2,2):1}, {(2,2):1})
    
    # Term 9
    u9 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    v9 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    w9 = {(0,0):-1, (1,0):-1, (0,1):-1, (1,1):-1}
    set_term(9, u9, v9, w9)
    
    # Term 10
    u10 = {(0,0):1, (0,2):1, (1,0):1, (1,2):1}
    v10 = {(0,0):1, (0,2):1, (1,0):1, (1,2):1}
    w10 = {(0,0):1, (2,0):-1, (0,1):1, (2,1):-1}
    set_term(10, u10, v10, w10)
    
    # Term 11
    u11 = {(0,1):1, (0,2):1, (1,1):1, (1,2):1}
    v11 = {(0,1):1, (0,2):1, (1,1):1, (1,2):1}
    w11 = {(1,0):1, (2,0):1, (1,1):1, (2,1):1}
    set_term(11, u11, v11, w11)
    
    # Term 12
    u12 = {(0,0):1, (0,1):1, (2,0):1, (2,1):1}
    v12 = {(0,0):1, (0,1):1, (2,0):1, (2,1):1}
    w12 = {(0,0):-1, (1,0):-1, (0,2):-1, (1,2):-1}
    set_term(12, u12, v12, w12)
    
    # Term 13
    u13 = {(0,0):1, (0,2):1, (2,0):1, (2,2):1}
    v13 = {(0,0):1, (0,2):1, (2,0):1, (2,2):1}
    w13 = {(0,0):1, (2,0):-1, (0,2):1, (2,2):-1}
    set_term(13, u13, v13, w13)
    
    # Term 14
    u14 = {(0,1):1, (0,2):1, (2,1):1, (2,2):1}
    v14 = {(0,1):1, (0,2):1, (2,1):1, (2,2):1}
    w14 = {(1,0):1, (2,0):1, (1,2):1, (2,2):1}
    set_term(14, u14, v14, w14)
    
    # Term 15
    u15 = {(1,0):1, (1,1):1, (2,0):1, (2,1):1}
    v15 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    w15 = {(0,1):-1, (1,1):-1, (0,2):-1, (1,2):-1}
    set_term(15, u15, v15, w15)
    
    # Term 16
    u16 = {(1,0):1, (1,2):1, (2,0):1, (2,2):1}
    v16 = {(0,0):1, (0,2):1, (1,0):1, (1,2):1}
    w16 = {(0,1):1, (2,1):-1, (0,2):1, (2,2):-1}
    set_term(16, u16, v16, w16)
    
    # Term 17
    u17 = {(1,1):1, (1,2):1, (2,1):1, (2,2):1}
    v17 = {(0,1):1, (0,2):1, (1,1):1, (1,2):1}
    w17 = {(1,1):1, (2,1):1, (1,2):1, (2,2):1}
    set_term(17, u17, v17, w17)
    
    # Term 18
    u18 = {(0,0):1, (1,0):1, (2,0):1}
    v18 = {(0,0):1, (1,0):1, (2,0):1}
    w18 = {(0,0):-1, (0,1):-1, (0,2):-1}
    set_term(18, u18, v18, w18)
    
    # Term 19
    u19 = {(0,0):1, (1,1):1, (2,2):1}
    v19 = {(0,0):1, (1,1):1, (2,2):1}
    w19 = {(0,0):1, (1,1):1, (2,2):1}
    set_term(19, u19, v19, w19)
    
    # Term 20
    u20 = {(0,1):1, (1,2):1, (2,0):1}
    v20 = {(0,1):1, (1,2):1, (2,0):1}
    w20 = {(1,0):-1, (2,1):-1, (0,2):1}
    set_term(20, u20, v20, w20)
    
    # Term 21
    u21 = {(0,2):1, (1,0):1, (2,1):1}
    v21 = {(0,2):1, (1,0):1, (2,1):1}
    w21 = {(2,0):-1, (0,1):1, (1,2):1}
    set_term(21, u21, v21, w21)
    
    # Term 22
    u22 = {(0,1):1, (1,0):1, (2,2):1}
    v22 = {(0,1):1, (1,0):1, (2,2):1}
    w22 = {(1,0):1, (0,1):1, (2,2):1}
    set_term(22, u22, v22, w22)
    
    return U, V, W


def _verify_3x3x3(U, V, W):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(23):
                    s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return get_schoolbook()
    
    # Get rank-23 decomposition for 3x3x3
    U3, V3, W3 = _get_lader_23()
    
    if _verify_3x3x3(U3, V3, W3):
        rank = 46
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # Block 1: k=0,1,2
        U[:, 0:23] = U3
        # V indices: j*6 + k. k in 0..2
        for j in range(3):
            for k in range(3):
                V[j*6 + k, 0:23] = V3[j*3 + k, :]
        # W indices: k*3 + i. k in 0..2
        for k in range(3):
            for i in range(3):
                W[k*3 + i, 0:23] = W3[k*3 + i, :]
                
        # Block 2: k=3,4,5
        U[:, 23:46] = U3
        # V indices: j*6 + k. k in 3..5 -> map to V3[j*3 + (k-3)]
        for j in range(3):
            for k in range(3, 6):
                V[j*6 + k, 23:46] = V3[j*3 + (k-3), :]
        # W indices: k*3 + i. k in 3..5 -> map to W3[(k-3)*3 + i]
        for k in range(3, 6):
            for i in range(3):
                W[k*3 + i, 23:46] = W3[(k-3)*3 + i, :]
                
        # Verify full 3x3x6 decomposition
        max_err = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    s = 0
                    for r in range(46):
                        s += U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
                    max_err = max(max_err, abs(s - 1))
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return get_schoolbook()
