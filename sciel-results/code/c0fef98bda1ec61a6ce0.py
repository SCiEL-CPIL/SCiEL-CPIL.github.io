import numpy as np

def construct():
    n, m, p = 2, 2, 11
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    
    # Process 5 blocks of 2 columns each using Strassen (rank 7)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B[j, k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices for C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1
        
    # Last column k=10: schoolbook (rank 4)
    k2 = 10
    v_e2 = k2      # B[0, 10]
    v_g2 = 11 + k2 # B[1, 10]
    w_c02 = 2 * k2      # C[0, 10]
    w_c12 = 2 * k2 + 1  # C[1, 10]
    
    # a * B[0, 10]
    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1
    
    # b * B[1, 10]
    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1
    
    # c * B[0, 10]
    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1
    
    # d * B[1, 10]
    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
