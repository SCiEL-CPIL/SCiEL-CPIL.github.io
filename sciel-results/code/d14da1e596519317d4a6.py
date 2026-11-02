import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 140s limit, 20s margin
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            add_strassen_block(0, j_start, k_start)
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
