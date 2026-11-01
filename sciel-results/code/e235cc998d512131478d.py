"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0 (Rank 2)
    # r=0: i=0
    U[0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        W[k * n + 0][0] = 1
        
    # r=1: i=1
    U[3][1] = 1
    for k in range(p):
        V[0 * p + k][1] = 1
        W[k * n + 1][1] = 1
        
    # Part 2: j in {1, 2} (Rank 28)
    # Split k into 4 blocks of size 2
    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        # V indices for B block q
        v_e = 1 * p + k0
        v_f = 1 * p + k1
        v_g = 2 * p + k0
        v_h = 2 * p + k1
        
        # W indices for C block q
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        # U indices for A_R
        u_a = 0 * m + 1 # A[0,1]
        u_b = 0 * m + 2 # A[0,2]
        u_c = 1 * m + 1 # A[1,1]
        u_d = 1 * m + 2 # A[1,2]
        
        # M1: (a+d)(e+h)
        r = 2 + 7 * q + 0
        U[u_a][r] += 1
        U[u_d][r] += 1
        V[v_e][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += 1
        W[w_c11][r] += 1
        
        # M2: (c+d)e
        r = 2 + 7 * q + 1
        U[u_c][r] += 1
        U[u_d][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1
        W[w_c11][r] -= 1
        
        # M3: a(f-h)
        r = 2 + 7 * q + 2
        U[u_a][r] += 1
        V[v_f][r] += 1
        V[v_g][r] -= 1
        W[w_c01][r] += 1
        W[w_c11][r] -= 1
        
        # M4: d(g-e)
        r = 2 + 7 * q + 3
        U[u_d][r] += 1
        V[v_g][r] += 1
        V[v_e][r] -= 1
        W[w_c00][r] -= 1
        W[w_c10][r] += 1
        
        # M5: (a+b)h
        r = 2 + 7 * q + 4
        U[u_a][r] += 1
        U[u_b][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += 1
        W[w_c01][r] += 1
        
        # M6: (c-a)(e+f)
        r = 2 + 7 * q + 5
        U[u_c][r] += 1
        U[u_a][r] -= 1
        V[v_e][r] += 1
        V[v_f][r] += 1
        W[w_c11][r] += 1
        
        # M7: (b-d)(g+h)
        r = 2 + 7 * q + 6
        U[u_b][r] += 1
        U[u_d][r] -= 1
        V[v_g][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] -= 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
