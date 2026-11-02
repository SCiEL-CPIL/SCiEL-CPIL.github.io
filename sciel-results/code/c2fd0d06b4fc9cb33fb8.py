DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank is 6 blocks * 7 terms = 42
    rank = 42
    
    # U: (n*m, rank) = (4, 42)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank) = (24, 42)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) = (24, 42)
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # u: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # v: coefficients for B[0,0], B[0,1], B[1,0], B[1,1] (local)
    # w: coefficients for C[0,0], C[0,1], C[1,0], C[1,1] (local)
    # The term contributes u*v to w
    
    # M1
    u1 = [1, 0, 0, 1] # A00 + A11
    v1 = [1, 0, 0, 1] # B00 + B11
    w1 = [1, 0, 0, -1] # C00 += 1, C11 += -1 (from C11 = ... - M5 + M7 ... wait)
    # Let's re-verify W mapping.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # w1[0] (C00) += 1
    # w1[3] (C11) += 1
    # w1[1] (C01) += 0
    # w1[2] (C10) += 0
    w1 = [1, 0, 0, 1]
    
    # M2
    u2 = [0, 0, 1, 1] # A10 + A11
    v2 = [1, 0, 0, 0] # B00
    # C10 += M2
    # C11 += -M2
    w2 = [0, 0, 1, -1]
    
    # M3
    u3 = [1, 0, 0, 0] # A00
    v3 = [0, 1, 0, -1] # B01 - B11
    # C01 += M3
    # C11 += M3
    w3 = [0, 1, 0, 1]
    
    # M4
    u4 = [0, 0, 0, 1] # A11
    v4 = [0, 0, 1, -1] # B10 - B00
    # C00 += M4
    # C10 += M4
    w4 = [1, 0, 1, 0]
    
    # M5
    u5 = [1, 1, 0, 0] # A00 + A01
    v5 = [0, 0, 0, 1] # B11
    # C00 += -M5
    # C01 += M5
    w5 = [-1, 1, 0, 0]
    
    # M6
    u6 = [0, 1, -1, 0] # A01 - A10
    v6 = [1, 1, 0, 0] # B00 + B01
    # C11 += M6
    w6 = [0, 0, 0, 1]
    
    # M7
    u7 = [-1, 0, 1, 0] # A10 - A00
    v7 = [0, 0, 1, 1] # B10 + B11
    # C00 += M7
    w7 = [1, 0, 0, 0]
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for s in range(6):
        for t in range(7):
            r = s * 7 + t
            u, v, w = terms[t]
            
            # Fill U
            # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for idx in range(4):
                U[idx][r] = u[idx]
                
            # Fill V
            # B indices: j in {0,1}, k_local in {0,1}
            # Global k = 2*s + k_local
            # V index = j * p + k_global
            # v[0] is B[0,0], v[1] is B[0,1], v[2] is B[1,0], v[3] is B[1,1]
            # Mapping:
            # v[0] -> j=0, kl=0 -> k=2s
            # v[1] -> j=0, kl=1 -> k=2s+1
            # v[2] -> j=1, kl=0 -> k=2s
            # v[3] -> j=1, kl=1 -> k=2s+1
            
            v_map = [
                (0, 2*s),      # v[0]
                (0, 2*s+1),    # v[1]
                (1, 2*s),      # v[2]
                (1, 2*s+1)     # v[3]
            ]
            for idx in range(4):
                j, k = v_map[idx]
                V[j * p + k][r] = v[idx]
                
            # Fill W
            # C indices: k_local in {0,1}, i in {0,1}
            # Global k = 2*s + k_local
            # W index = k_global * n + i
            # w[0] is C[0,0] -> i=0, kl=0 -> k=2s
            # w[1] is C[0,1] -> i=0, kl=1 -> k=2s+1
            # w[2] is C[1,0] -> i=1, kl=0 -> k=2s
            # w[3] is C[1,1] -> i=1, kl=1 -> k=2s+1
            
            w_map = [
                (2*s, 0),      # w[0]
                (2*s+1, 0),    # w[1]
                (2*s, 1),      # w[2]
                (2*s+1, 1)     # w[3]
            ]
            for idx in range(4):
                k, i = w_map[idx]
                W[k * n + i][r] = w[idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
