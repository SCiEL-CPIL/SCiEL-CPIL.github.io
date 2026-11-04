"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    # i0, k0, j0 are the top-left indices of the 2x2x2 block
    def add_strassen(i0, k0, j0, start_col):
        nonlocal col
        # Local indices
        # A: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        # B: (k0, j0), (k0, j0+1), (k0+1, j0), (k0+1, j0+1)
        # C: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        
        # Map to global indices
        # U index: i * m + k
        u_a = i0 * m + k0
        u_b = i0 * m + (k0 + 1)
        u_c = (i0 + 1) * m + k0
        u_d = (i0 + 1) * m + (k0 + 1)
        
        # V index: k * p + j
        v_e = k0 * p + j0
        v_f = k0 * p + (j0 + 1)
        v_g = (k0 + 1) * p + j0
        v_h = (k0 + 1) * p + (j0 + 1)
        
        # W index: j * n + i  (Note: prompt says W[k*n+i], but in seed it was W[k*n+i] where k was col of C and i was row of C? 
        # Seed: W[k * n + i]. k was the 3rd loop var (col of B / col of C). i was 1st loop var (row of A / row of C).
        # So W index is (col_C * n + row_C).
        # My j is col_C. My i is row_C.
        # So W index is j * n + i.
        
        w_p = j0 * n + i0
        w_q = (j0 + 1) * n + i0
        w_r = j0 * n + (i0 + 1)
        w_s = (j0 + 1) * n + (i0 + 1)
        
        # 1: (a+d)(e+h) -> p, s
        U[u_a][start_col] += 1
        U[u_d][start_col] += 1
        V[v_e][start_col] += 1
        V[v_h][start_col] += 1
        W[w_p][start_col] += 1
        W[w_s][start_col] += 1
        col += 1
        
        # 2: (c+d)e -> r, -s
        U[u_c][start_col] += 1
        U[u_d][start_col] += 1
        V[v_e][start_col] += 1
        W[w_r][start_col] += 1
        W[w_s][start_col] -= 1
        col += 1
        
        # 3: a(f-h) -> q, s
        U[u_a][start_col] += 1
        V[v_f][start_col] += 1
        V[v_h][start_col] -= 1
        W[w_q][start_col] += 1
        W[w_s][start_col] += 1
        col += 1
        
        # 4: d(g-e) -> p, r
        U[u_d][start_col] += 1
        V[v_g][start_col] += 1
        V[v_e][start_col] -= 1
        W[w_p][start_col] += 1
        W[w_r][start_col] += 1
        col += 1
        
        # 5: (a+b)h -> -p, q
        U[u_a][start_col] += 1
        U[u_b][start_col] += 1
        V[v_h][start_col] += 1
        W[w_p][start_col] -= 1
        W[w_q][start_col] += 1
        col += 1
        
        # 6: (c-a)(e+f) -> s
        U[u_c][start_col] += 1
        U[u_a][start_col] -= 1
        V[v_e][start_col] += 1
        V[v_f][start_col] += 1
        W[w_s][start_col] += 1
        col += 1
        
        # 7: (b-d)(g+h) -> p
        U[u_b][start_col] += 1
        U[u_d][start_col] -= 1
        V[v_g][start_col] += 1
        V[v_h][start_col] += 1
        W[w_p][start_col] += 1
        col += 1

    # Helper to add Schoolbook 1x2x2 block
    def add_schoolbook_1x2x2(i0, k0, j0, start_col):
        nonlocal col
        # A: (i0, k0), (i0, k0+1)
        u_a = i0 * m + k0
        u_b = i0 * m + (k0 + 1)
        
        # B: (k0, j0), (k0, j0+1), (k0+1, j0), (k0+1, j0+1)
        v_c = k0 * p + j0
        v_d = k0 * p + (j0 + 1)
        v_e = (k0 + 1) * p + j0
        v_f = (k0 + 1) * p + (j0 + 1)
        
        # C: (i0, j0), (i0, j0+1)
        w_u = j0 * n + i0
        w_v = (j0 + 1) * n + i0
        
        # 1: a*c -> u
        U[u_a][start_col] += 1
        V[v_c][start_col] += 1
        W[w_u][start_col] += 1
        col += 1
        
        # 2: b*e -> u
        U[u_b][start_col] += 1
        V[v_e][start_col] += 1
        W[w_u][start_col] += 1
        col += 1
        
        # 3: a*d -> v
        U[u_a][start_col] += 1
        V[v_d][start_col] += 1
        W[w_v][start_col] += 1
        col += 1
        
        # 4: b*f -> v
        U[u_b][start_col] += 1
        V[v_f][start_col] += 1
        W[w_v][start_col] += 1
        col += 1

    # Apply blocks
    # 4 Strassen blocks (2x2x2)
    add_strassen(0, 0, 0, col)
    add_strassen(0, 0, 2, col)
    add_strassen(0, 2, 0, col)
    add_strassen(0, 2, 2, col)
    
    # 4 Schoolbook blocks (1x2x2)
    add_schoolbook_1x2x2(2, 0, 0, col)
    add_schoolbook_1x2x2(2, 0, 2, col)
    add_schoolbook_1x2x2(2, 2, 0, col)
    add_schoolbook_1x2x2(2, 2, 2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
