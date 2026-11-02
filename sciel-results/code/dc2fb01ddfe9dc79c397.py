import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin
    
    n, m, p = 3, 4, 6
    terms = []
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Strassen 2x2x2: A[i1:i3, j1:j3] * B[j1:j3, k1:k3] -> C[i1:i3, k1:k3]"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        # M1 = (a+d)(e+h)
        terms.append((
            {a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}
        ))
        # M2 = (c+d)e
        terms.append((
            {c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}
        ))
        # M3 = a(f-h)
        terms.append((
            {a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}
        ))
        # M4 = d(g-e)
        terms.append((
            {d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}
        ))
        # M5 = (a+b)h
        terms.append((
            {a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}
        ))
        # M6 = (c-a)(e+f)
        terms.append((
            {c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}
        ))
        # M7 = (b-d)(g+h)
        terms.append((
            {b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}
        ))
    
    def add_schoolbook(i, j, k):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))
    
    # Row 0: schoolbook (4 middle * 6 right = 24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    # Split m=4 into two 2-col groups, p=6 into three 2-col groups
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        if not time_ok():
            break
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
