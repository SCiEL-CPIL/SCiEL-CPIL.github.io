import numpy as np

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for u in range(2):
        for v in range(4):
            base = 7 * (4 * u + v)
            
            a00 = 0 * m + 2 * u
            a01 = 0 * m + 2 * u + 1
            a10 = 1 * m + 2 * u
            a11 = 1 * m + 2 * u + 1
            
            b00 = (2 * u) * p + 2 * v
            b01 = (2 * u) * p + 2 * v + 1
            b10 = (2 * u + 1) * p + 2 * v
            b11 = (2 * u + 1) * p + 2 * v + 1
            
            c00 = (2 * v) * n + 0
            c01 = (2 * v + 1) * n + 0
            c10 = (2 * v) * n + 1
            c11 = (2 * v + 1) * n + 1
            
            U[a00, base] += 1
            U[a11, base] += 1
            V[b00, base] += 1
            V[b11, base] += 1
            W[c00, base] += 1
            W[c11, base] += 1
            
            U[a10, base + 1] += 1
            U[a11, base + 1] += 1
            V[b00, base + 1] += 1
            W[c10, base + 1] += 1
            W[c11, base + 1] -= 1
            
            U[a00, base + 2] += 1
            V[b01, base + 2] += 1
            V[b11, base + 2] -= 1
            W[c01, base + 2] += 1
            W[c11, base + 2] += 1
            
            U[a11, base + 3] += 1
            V[b10, base + 3] += 1
            V[b00, base + 3] -= 1
            W[c00, base + 3] += 1
            W[c10, base + 3] += 1
            
            U[a00, base + 4] += 1
            U[a01, base + 4] += 1
            V[b11, base + 4] += 1
            W[c00, base + 4] -= 1
            W[c01, base + 4] += 1
            
            U[a10, base + 5] += 1
            U[a00, base + 5] -= 1
            V[b00, base + 5] += 1
            V[b01, base + 5] += 1
            W[c11, base + 5] += 1
            
            U[a01, base + 6] += 1
            U[a11, base + 6] -= 1
            V[b10, base + 6] += 1
            V[b11, base + 6] += 1
            W[c00, base + 6] += 1
            
            col += 7
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
