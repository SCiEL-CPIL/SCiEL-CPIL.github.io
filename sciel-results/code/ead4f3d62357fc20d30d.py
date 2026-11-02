import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    terms = []
    
    def add_term(u_sparse, v_sparse, w_sparse):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for idx, val in u_sparse.items():
            u[idx] = val
        for idx, val in v_sparse.items():
            v[idx] = val
        for idx, val in w_sparse.items():
            w[idx] = val
        terms.append((u, v, w))

    def strassen_2x2(i0, j0, k0):
        # i0, j0, k0 are the starting indices for I, J, K
        # Block is 2x2x2
        # I: i0, i0+1
        # J: j0, j0+1
        # K: k0, k0+1
        
        # M1 = (A11+A22)(B11+B22)
        u = {i0*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v = {j0*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: 1, (k0+1)*n + i0: 1, k0*n + (i0+1): 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # M2 = (A21+A22)B11
        u = {(i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v = {j0*p + k0: 1}
        w = {k0*n + (i0+1): 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # M3 = A11(B12-B22)
        u = {i0*m + j0: 1}
        v = {j0*p + (k0+1): 1, (j0+1)*p + (k0+1): -1}
        w = {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # M4 = A22(B21-B11)
        u = {(i0+1)*m + (j0+1): 1}
        v = {(j0+1)*p + k0: 1, j0*p + k0: -1}
        w = {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # M5 = (A11+A12)B22
        u = {i0*m + j0: 1, i0*m + (j0+1): 1}
        v = {(j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: -1, (k0+1)*n + i0: 1}
        add_term(u, v, w)
        
        # M6 = (A21-A11)(B11+B12)
        u = {(i0+1)*m + j0: 1, i0*m + j0: -1}
        v = {j0*p + k0: 1, j0*p + (k0+1): 1}
        w = {(k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # M7 = (A12-A22)(B21+B22)
        u = {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): -1}
        v = {(j0+1)*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: 1, k0*n + (i0+1): 1}
        add_term(u, v, w)

    def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {i*m + j: 1}
                    v = {j*p + k: 1}
                    w = {k*n + i: 1}
                    add_term(u, v, w)

    # Block 1: I in {0,1}, J in {0,1}, K in {0,1} -> Strassen
    strassen_2x2(0, 0, 0)
    
    # Block 2: I in {0,1}, J in {0,1}, K in {2,3,4} -> Schoolbook (2x2x3)
    schoolbook_block(0, 2, 0, 2, 2, 5)
    
    # Block 3: I in {2}, J in {0,1}, K in {0,1} -> Schoolbook (1x2x2)
    schoolbook_block(2, 3, 0, 2, 0, 2)
    
    # Block 4: I in {2}, J in {0,1}, K in {2,3,4} -> Schoolbook (1x2x3)
    schoolbook_block(2, 3, 0, 2, 2, 5)
    
    # Block 5: I in {0,1}, J in {2,3}, K in {0,1} -> Strassen
    strassen_2x2(0, 2, 0)
    
    # Block 6: I in {0,1}, J in {2,3}, K in {2,3,4} -> Schoolbook (2x2x3)
    schoolbook_block(0, 2, 2, 4, 2, 5)
    
    # Block 7: I in {0,1}, J in {4}, K in {0,1} -> Schoolbook (2x1x2)
    schoolbook_block(0, 2, 4, 5, 0, 2)
    
    # Block 8: I in {0,1}, J in {4}, K in {2,3,4} -> Schoolbook (2x1x3)
    schoolbook_block(0, 2, 4, 5, 2, 5)
    
    # Block 9: I in {2}, J in {2,3,4}, K in {0,1} -> Schoolbook (1x3x2)
    schoolbook_block(2, 3, 2, 5, 0, 2)
    
    # Block 10: I in {2}, J in {2,3,4}, K in {2,3,4} -> Schoolbook (1x3x3)
    schoolbook_block(2, 3, 2, 5, 2, 5)
    
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            if abs(U_list[i][j]) < 1e-10:
                U_list[i][j] = 0
            else:
                U_list[i][j] = round(U_list[i][j])
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            if abs(V_list[i][j]) < 1e-10:
                V_list[i][j] = 0
            else:
                V_list[i][j] = round(V_list[i][j])
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            if abs(W_list[i][j]) < 1e-10:
                W_list[i][j] = 0
            else:
                W_list[i][j] = round(W_list[i][j])
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

