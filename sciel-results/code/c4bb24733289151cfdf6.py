import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10
    
    R = 29
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((np_, R))
    
    col = 0
    
    # Strassen terms for the 2x2x2 sub-block (i,j,k in {0,1})
    # A indices: a00=0, a01=1, a10=3, a11=4
    # B indices: b00=0, b01=1, b10=5, b11=6
    # C indices: c00=0, c01=1, c10=2, c11=3
    
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        ({0: 1, 4: 1}, {0: 1, 6: 1}, {0: 1, 3: 1}),
        # M2 = (a10+a11)b00
        ({3: 1, 4: 1}, {0: 1}, {2: 1}),
        # M3 = a00(b01-b11)
        ({0: 1}, {1: 1, 6: -1}, {1: 1}),
        # M4 = a11(b10-b00)
        ({4: 1}, {5: 1, 0: -1}, {2: 1, 0: 1}),
        # M5 = (a00+a01)b11
        ({0: 1, 1: 1}, {6: 1}, {0: -1, 1: 1}),
        # M6 = (a10-a00)(b00+b01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a01-a11)(b10+b11)
        ({1: 1, 4: -1}, {5: 1, 6: 1}, {0: 1}),
    ]
    
    for u_dict, v_dict, w_dict in strassen_terms:
        for a, val in u_dict.items():
            U[a, col] = val
        for b, val in v_dict.items():
            V[b, col] = val
        for c, val in w_dict.items():
            W[c, col] = val
        col += 1
    
    # Schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    assert col == 29, f"Expected 29, got {col}"
    
    # Verify
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    if not np.allclose(T_calc, T_target, atol=1e-9):
        # Fallback to schoolbook
        R_school = n * m * p
        U_s = np.zeros((nm, R_school))
        V_s = np.zeros((mp, R_school))
        W_s = np.zeros((np_, R_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col_s] = 1
                    V_s[j*p+k, col_s] = 1
                    W_s[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
