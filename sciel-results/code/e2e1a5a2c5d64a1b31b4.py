"""Matrix multiplication 4x4x6 with rank-84 decomposition via Strassen on 2x2x2 blocks."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    
    def strassen_2x2x2(ai, aj, bk):
        nonlocal col
        u_a = ai * m + aj
        u_b = ai * m + (aj + 1)
        u_c = (ai + 1) * m + aj
        u_d = (ai + 1) * m + (aj + 1)
        v_e = aj * p + bk
        v_f = aj * p + (bk + 1)
        v_g = (aj + 1) * p + bk
        v_h = (aj + 1) * p + (bk + 1)
        w_p = bk * n + ai
        w_q = (bk + 1) * n + ai
        w_r = bk * n + (ai + 1)
        w_s = (bk + 1) * n + (ai + 1)
        
        # M1 = (a+d)(e+h)
        U[u_a][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1; V[v_h][col] = 1
        W[w_p][col] = 1; W[w_s][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[u_c][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1
        W[w_r][col] = 1; W[w_s][col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[u_a][col] = 1
        V[v_f][col] = 1; V[v_h][col] = -1
        W[w_q][col] = 1; W[w_s][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[u_d][col] = 1
        V[v_g][col] = 1; V[v_e][col] = -1
        W[w_q][col] = -1; W[w_r][col] = 1; W[w_s][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[u_a][col] = 1; U[u_b][col] = 1
        V[v_h][col] = 1
        W[w_p][col] = -1; W[w_q][col] = 1
        col += 1
        
        # M6 = (c-d)(f+h)
        U[u_c][col] = 1; U[u_d][col] = -1
        V[v_f][col] = 1; V[v_h][col] = 1
        W[w_q][col] = 1; W[w_r][col] = -1
        col += 1
        
        # M7 = (b-d)(e+g)
        U[u_b][col] = 1; U[u_d][col] = -1
        V[v_e][col] = 1; V[v_g][col] = 1
        W[w_p][col] = 1; W[w_r][col] = -1
        col += 1
    
    for ai in range(0, 4, 2):
        for aj in range(0, 4, 2):
            for bk in range(0, 6, 2):
                strassen_2x2x2(ai, aj, bk)
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
