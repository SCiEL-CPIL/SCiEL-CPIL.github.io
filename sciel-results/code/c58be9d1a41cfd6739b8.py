"""
5x5 matrix multiplication - rank 124 block decomposition (2+3 split).
Uses Strassen for 2x2 block and schoolbook for others.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    # Block indices
    # A: i (0..4), j (0..4)
    # B: j (0..4), k (0..4)
    # C: i (0..4), k (0..4)
    
    # Split: 0,1 are "small" (2), 2,3,4 are "large" (3)
    small_idx = [0, 1]
    large_idx = [2, 3, 4]
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i,j): val}
        v_vals: dict {(j,k): val}
        w_vals: dict {(k,i): val}
        """
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)
        
        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val
            
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # A_22: i in 0,1; j in 0,1
    # B_22: j in 0,1; k in 0,1
    # C_22: i in 0,1; k in 0,1
    # Use Strassen for A_22 B_22
    
    # Strassen terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    u = {(0,0):1, (1,1):1}
    v = {(0,0):1, (1,1):1}
    w = {(0,0):1, (1,1):1}
    add_term(u, v, w)
    
    # M2 = (A10+A11)B00
    u = {(1,0):1, (1,1):1}
    v = {(0,0):1}
    w = {(0,1):1, (1,1):-1}
    add_term(u, v, w)
    
    # M3 = A00(B01-B11)
    u = {(0,0):1}
    v = {(0,1):1, (1,1):-1}
    w = {(1,0):1, (1,1):1}
    add_term(u, v, w)
    
    # M4 = A11(B10-B00)
    u = {(1,1):1}
    v = {(1,0):1, (0,0):-1}
    w = {(0,0):1, (0,1):1}
    add_term(u, v, w)
    
    # M5 = (A00+A01)B11
    u = {(0,0):1, (0,1):1}
    v = {(1,1):1}
    w = {(0,0):-1, (1,0):1}
    add_term(u, v, w)
    
    # M6 = (A10-A00)(B00+B01)
    u = {(1,0):1, (0,0):-1}
    v = {(0,0):1, (0,1):1}
    w = {(1,1):1}
    add_term(u, v, w)
    
    # M7 = (A01-A11)(B10+B11)
    u = {(0,1):1, (1,1):-1}
    v = {(1,0):1, (1,1):1}
    w = {(0,0):1}
    add_term(u, v, w)
    
    # A_23 B_32 schoolbook
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 2. C_23 = A_22 B_23 + A_23 B_33
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 3. C_32 = A_32 B_22 + A_33 B_32
    for i in [2, 3, 4]:
        for j in [0, 1]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    for i in [2, 3, 4]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 4. C_33 = A_32 B_23 + A_33 B_33
    for i in [2, 3, 4]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    for i in [2, 3, 4]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
