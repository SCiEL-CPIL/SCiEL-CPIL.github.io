import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # U is 4 x R
    # V is 26 x R
    # W is 26 x R
    
    # We will build lists of columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a column
    def add_factor(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen factors for 2x2 x 2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices for a pair (k1, k2):
    # e = B[0, k1] -> idx 0*p + k1
    # f = B[0, k2] -> idx 0*p + k2
    # g = B[1, k1] -> idx 1*p + k1
    # h = B[1, k2] -> idx 1*p + k2
    # C indices for a pair (k1, k2):
    # C[0, k1] -> idx k1*n + 0
    # C[1, k1] -> idx k1*n + 1
    # C[0, k2] -> idx k2*n + 0
    # C[1, k2] -> idx k2*n + 1

    def get_strassen_factors(k1, k2):
        factors = []
        
        # M1 = (a+d)(e+h)
        u1 = [1, 0, 0, 1]
        v1 = [0] * (m * p)
        v1[0 * p + k1] = 1
        v1[1 * p + k2] = 1
        w1 = [0] * (p * n)
        w1[k1 * n + 0] = 1
        w1[k2 * n + 1] = 1
        factors.append((u1, v1, w1))
        
        # M2 = (c+d)e
        u2 = [0, 0, 1, 1]
        v2 = [0] * (m * p)
        v2[0 * p + k1] = 1
        w2 = [0] * (p * n)
        w2[k1 * n + 1] = 1
        w2[k2 * n + 1] = -1
        factors.append((u2, v2, w2))
        
        # M3 = a(f-h)
        u3 = [1, 0, 0, 0]
        v3 = [0] * (m * p)
        v3[0 * p + k2] = 1
        v3[1 * p + k2] = -1
        w3 = [0] * (p * n)
        w3[k2 * n + 0] = 1
        w3[k2 * n + 1] = 1
        factors.append((u3, v3, w3))
        
        # M4 = d(g-e)
        u4 = [0, 0, 0, 1]
        v4 = [0] * (m * p)
        v4[1 * p + k1] = 1
        v4[0 * p + k1] = -1
        w4 = [0] * (p * n)
        w4[k1 * n + 0] = 1
        w4[k1 * n + 1] = 1
        factors.append((u4, v4, w4))
        
        # M5 = (a+b)h
        u5 = [1, 1, 0, 0]
        v5 = [0] * (m * p)
        v5[1 * p + k2] = 1
        w5 = [0] * (p * n)
        w5[k1 * n + 0] = -1
        w5[k2 * n + 0] = 1
        factors.append((u5, v5, w5))
        
        # M6 = (c-a)(e+f)
        u6 = [-1, 0, 1, 0]
        v6 = [0] * (m * p)
        v6[0 * p + k1] = 1
        v6[0 * p + k2] = 1
        w6 = [0] * (p * n)
        w6[k2 * n + 1] = 1
        factors.append((u6, v6, w6))
        
        # M7 = (b-d)(g+h)
        u7 = [0, 1, 0, -1]
        v7 = [0] * (m * p)
        v7[1 * p + k1] = 1
        v7[1 * p + k2] = 1
        w7 = [0] * (p * n)
        w7[k1 * n + 0] = 1
        factors.append((u7, v7, w7))
        
        return factors

    # Process pairs
    num_pairs = p // 2
    for i in range(num_pairs):
        k1 = 2 * i
        k2 = 2 * i + 1
        factors = get_strassen_factors(k1, k2)
        for u, v, w in factors:
            add_factor(u, v, w)
            
    # Process remaining single column
    if p % 2 != 0:
        k = 2 * num_pairs
        # Schoolbook for 2x2 x 1
        # C[0, k] = a B[0, k] + b B[1, k]
        # C[1, k] = c B[0, k] + d B[1, k]
        
        # a B[0, k]
        u = [1, 0, 0, 0]
        v = [0] * (m * p)
        v[0 * p + k] = 1
        w = [0] * (p * n)
        w[k * n + 0] = 1
        add_factor(u, v, w)
        
        # b B[1, k]
        u = [0, 1, 0, 0]
        v = [0] * (m * p)
        v[1 * p + k] = 1
        w = [0] * (p * n)
        w[k * n + 0] = 1
        add_factor(u, v, w)
        
        # c B[0, k]
        u = [0, 0, 1, 0]
        v = [0] * (m * p)
        v[0 * p + k] = 1
        w = [0] * (p * n)
        w[k * n + 1] = 1
        add_factor(u, v, w)
        
        # d B[1, k]
        u = [0, 0, 0, 1]
        v = [0] * (m * p)
        v[1 * p + k] = 1
        w = [0] * (p * n)
        w[k * n + 1] = 1
        add_factor(u, v, w)

    # Convert to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
