import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8
    
    # Strassen 2x2 forms
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    # C00 = m0 - m3 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    
    x = np.array([
        [1, 0, 0, 1],  # a00+a11
        [0, 0, 1, 1],  # a10+a11
        [1, 0, 0, 0],  # a00
        [0, 0, 0, 1],  # a11
        [1, 1, 0, 0],  # a00+a01
        [-1, 0, 1, 0], # a10-a00
        [0, 1, 0, -1]  # a01-a11
    ], dtype=int)
    
    y = np.array([
        [1, 0, 0, 1],  # b00+b11
        [1, 0, 0, 0],  # b00
        [0, 1, 0, -1], # b01-b11
        [-1, 0, 1, 0], # b10-b00
        [0, 0, 0, 1],  # b11
        [1, 1, 0, 0],  # b00+b01
        [0, 0, 1, 1]   # b10+b11
    ], dtype=int)
    
    z = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00: m0-m3+m6
        [0, 0, 1, 0, 1, 0, 0],   # c01: m2+m4
        [0, 1, 0, 1, 0, 0, 0],   # c10: m1+m3
        [1, -1, 1, 0, 0, 1, 0]   # c11: m0-m1+m2+m5
    ], dtype=int)
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    r = 0
    for ab in range(2):
        for bg in range(4):
            for s in range(7):
                for i in range(n):
                    for jp in range(2):
                        U[i*m + ab*2 + jp, r] = x[s, i*2 + jp]
                for jp in range(2):
                    for kp in range(2):
                        V[(ab*2+jp)*p + 2*bg + kp, r] = y[s, jp*2 + kp]
                for i in range(n):
                    for kp in range(2):
                        W[(2*bg+kp)*n + i, r] = z[i*2+kp, s]
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
