import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    # u_idx: (i, j) -> i*m + j
    # v_idx: (j, k) -> j*p + k
    # w_idx: (k, i) -> k*n + i
    
    col = 0
    for l in range(4):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Strassen part for j=0,1
        # M1 = (a+d)(e+h)
        # U: a, d. V: e, h. W: C00, C11
        # a: (0,0), d: (1,1)
        # e: (0, k0), h: (1, k1)
        # C00: (k0, 0), C11: (k1, 1)
        
        # Term 0: M1
        U[0*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # Term 1: M2 = (c+d)e
        # c: (1,0), d: (1,1)
        # e: (0, k0)
        # C10: (k0, 1)
        U[1*m + 0, col] = 1
        U[1*m + 1, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        # Term 2: M3 = a(f-h)
        # a: (0,0)
        # f: (0, k1), h: (1, k1)
        # C01: (k1, 0)
        U[0*m + 0, col] = 1
        V[0*p + k1, col] = 1
        V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        
        # Term 3: M4 = d(g-e)
        # d: (1,1)
        # g: (1, k0), e: (0, k0)
        # C00: (k0, 0), C10: (k0, 1)
        U[1*m + 1, col] = 1
        V[1*p + k0, col] = 1
        V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        # Term 4: M5 = (a+b)h
        # a: (0,0), b: (0,1)
        # h: (1, k1)
        # C00: (k0, 0), C01: (k1, 0)
        U[0*m + 0, col] = 1
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        
        # Term 5: M6 = (c-d)(f+h)
        # c: (1,0), d: (1,1)
        # f: (0, k1), h: (1, k1)
        # C11: (k1, 1)
        U[1*m + 0, col] = 1
        U[1*m + 1, col] = -1
        V[0*p + k1, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # Term 6: M7 = (b-c)(e+g)
        # b: (0,1), c: (1,0)
        # e: (0, k0), g: (1, k0)
        # C00: (k0, 0)
        U[0*m + 1, col] = 1
        U[1*m + 0, col] = -1
        V[0*p + k0, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
        
        # Trivial part for j=2
        # A_{0,2} B_{2, k0}
        U[0*m + 2, col] = 1
        V[2*p + k0, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
        
        # A_{0,2} B_{2, k1}
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k1*n + 0, col] = 1
        col += 1
        
        # A_{1,2} B_{2, k0}
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        # A_{1,2} B_{2, k1}
        U[1*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
