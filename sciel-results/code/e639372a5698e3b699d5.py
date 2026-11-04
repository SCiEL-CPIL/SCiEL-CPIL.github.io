import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # rank = 68
    rank = 68
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeff, v_coeff, w_coeff):
        nonlocal col
        # u_coeff: dict {(i,j): val}
        # v_coeff: dict {(j,k): val}
        # w_coeff: dict {(k,i): val}
        for (i, j), val in u_coeff.items():
            U[i * m + j][col] = val
        for (j, k), val in v_coeff.items():
            V[j * p + k][col] = val
        for (k, i), val in w_coeff.items():
            W[k * n + i][col] = val
        col += 1

    # Strassen vectors for 2x2x2
    # u, v, w are length 4 (flattened 2x2)
    # A: a,b,c,d -> 0,1,2,3
    # B: e,f,g,h -> 0,1,2,3
    # C: C00,C01,C10,C11 -> 0,1,2,3
    
    S_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for b in range(4):
        if time.time() > deadline:
            break
        # Block b corresponds to k in [2b, 2b+1]
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # 1. Strassen part (7 terms)
        # A11: i in {0,1}, j in {0,1}
        # B11: j in {0,1}, k' in {0,1} -> k in {k0, k1}
        # C_top: i in {0,1}, k' in {0,1} -> k in {k0, k1}
        
        for r in range(7):
            u_coeff = {}
            v_coeff = {}
            w_coeff = {}
            
            # Map local 2x2 indices to global
            # A: local (li, lj) -> global (li, lj)
            # B: local (lj, lk) -> global (lj, k0+lk)
            # C: local (li, lk) -> global (k0+lk, li)
            
            for li in range(2):
                for lj in range(2):
                    val = S_u[r][li * 2 + lj]
                    if val != 0:
                        u_coeff[(li, lj)] = val
            
            for lj in range(2):
                for lk in range(2):
                    val = S_v[r][lj * 2 + lk]
                    if val != 0:
                        k_global = k0 + lk
                        v_coeff[(lj, k_global)] = val
            
            for li in range(2):
                for lk in range(2):
                    val = S_w[r][li * 2 + lk]
                    if val != 0:
                        k_global = k0 + lk
                        w_coeff[(k_global, li)] = val
            
            add_term(u_coeff, v_coeff, w_coeff)
            
        # 2. A12 B21 part (4 terms)
        # A12: i in {0,1}, j=2
        # B21: j=2, k' in {0,1}
        # C_top: i in {0,1}, k' in {0,1}
        
        # P1: a02 b20 -> C00
        add_term({(0, 2): 1}, {(2, k0): 1}, {(k0, 0): 1})
        # P2: a02 b21 -> C01
        add_term({(0, 2): 1}, {(2, k1): 1}, {(k1, 0): 1})
        # P3: a12 b20 -> C10
        add_term({(1, 2): 1}, {(2, k0): 1}, {(k0, 1): 1})
        # P4: a12 b21 -> C11
        add_term({(1, 2): 1}, {(2, k1): 1}, {(k1, 1): 1})
        
        # 3. A21 B11 part (4 terms)
        # A21: i=2, j in {0,1}
        # B11: j in {0,1}, k' in {0,1}
        # C_bot: i=2, k' in {0,1}
        
        # Q1: a20 b00 -> C20
        add_term({(2, 0): 1}, {(0, k0): 1}, {(k0, 2): 1})
        # Q2: a20 b01 -> C21
        add_term({(2, 0): 1}, {(0, k1): 1}, {(k1, 2): 1})
        # Q3: a21 b10 -> C20
        add_term({(2, 1): 1}, {(1, k0): 1}, {(k0, 2): 1})
        # Q4: a21 b11 -> C21
        add_term({(2, 1): 1}, {(1, k1): 1}, {(k1, 2): 1})
        
        # 4. A22 B21 part (2 terms)
        # A22: i=2, j=2
        # B21: j=2, k' in {0,1}
        # C_bot: i=2, k' in {0,1}
        
        # R1: a22 b20 -> C20
        add_term({(2, 2): 1}, {(2, k0): 1}, {(k0, 2): 1})
        # R2: a22 b21 -> C21
        add_term({(2, 2): 1}, {(2, k1): 1}, {(k1, 2): 1})

    return {"U": U, "V": V, "W": W}
