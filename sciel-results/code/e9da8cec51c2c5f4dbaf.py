import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    # Part 1: j=2 part (2x1x9), schoolbook: 18 terms
    for i in range(n):
        for k in range(p):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + 2] = 1
            v[2 * p + k] = 1
            w[k * n + i] = 1
            add_term(u, v, w)
    
    # Part 2: j in {0,1} part (2x2x9)
    # Split k into 4 pairs + 1 single
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    k_single = 8
    
    for k1, k2 in k_pairs:
        # m1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[0] += 1; u[4] += 1
        v[k1] += 1; v[9 + k2] += 1
        w[k1 * 2 + 0] += 1; w[k2 * 2 + 1] += 1
        add_term(u, v, w)
        
        # m2: (A[1,0]+A[1,1])*B[0,0]
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[3] += 1; u[4] += 1
        v[k1] += 1
        w[k1 * 2 + 1] += 1; w[k2 * 2 + 1] += -1
        add_term(u, v, w)
        
        # m3: A[0,0]*(B[0,1]-B[1,1])
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[0] += 1
        v[k2] += 1; v[9 + k2] += -1
        w[k2 * 2 + 0] += 1; w[k2 * 2 + 1] += 1
        add_term(u, v, w)
        
        # m4: A[1,1]*(B[1,0]-B[0,0])
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[4] += 1
        v[9 + k1] += 1; v[k1] += -1
        w[k1 * 2 + 0] += 1; w[k1 * 2 + 1] += 1
        add_term(u, v, w)
        
        # m5: (A[0,0]+A[0,1])*B[1,1]
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[0] += 1; u[1] += 1
        v[9 + k2] += 1
        w[k1 * 2 + 0] += -1; w[k2 * 2 + 0] += 1
        add_term(u, v, w)
        
        # m6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[3] += 1; u[0] += -1
        v[k1] += 1; v[k2] += 1
        w[k2 * 2 + 1] += 1
        add_term(u, v, w)
        
        # m7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[1] += 1; u[4] += -1
        v[9 + k1] += 1; v[9 + k2] += 1
        w[k1 * 2 + 0] += 1
        add_term(u, v, w)
    
    # k=8 single: 2x2x1, schoolbook: 4 terms
    for i in range(n):
        for j in range(2):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * 3 + j] = 1
            v[j * 9 + k_single] = 1
            w[k_single * 2 + i] = 1
            add_term(u, v, w)
    
    rank = len(U_cols)
    
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
