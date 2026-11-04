import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We will build U, V, W as lists of lists
    # We don't know the rank ahead of time easily, so we'll append columns
    # Or we can estimate max rank and fill.
    # Max rank 40.
    
    U = []
    V = []
    W = []
    
    def add_rank(u_vec, v_vec, w_vec):
        # u_vec: list of length n*m
        # v_vec: list of length m*p
        # w_vec: list of length p*n
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    def init_vecs():
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        return u, v, w

    # Strassen terms for 2x2x2
    # Local indices i,j in 0..1 for A, j,k in 0..1 for B, i,k in 0..1 for C
    # A: i,j. B: j,k. C: i,k.
    # Strassen products P1..P7
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each P, we define U, V, and the contribution to W (which is the coefficient in C)
    # Actually, W for a specific rank r is the vector of coefficients of P_r in C.
    
    strassen_terms = [
        # P1
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # P2
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1}
        },
        # P3
        {
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1}
        },
        # P4
        {
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1}
        },
        # P5
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1}
        },
        # P6
        {
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1}
        },
        # P7
        {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1}
        }
    ]
    
    def add_strassen(a_j_off, b_j_off, b_k_off):
        # a_j_off: offset for A columns (0 or 2)
        # b_j_off: offset for B rows (0 or 2)
        # b_k_off: offset for B columns (0, 2, 4)
        
        for term in strassen_terms:
            u, v, w = init_vecs()
            # Fill U
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            # Fill V
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            # Fill W
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    def add_trivial_2x2x1(a_j_off, b_j_off, b_k_off):
        # 2x2x1 trivial rank 4
        # C_i0 = A_i0 B_00 + A_i1 B_10
        # Terms:
        # 1. A_00 B_00 -> C_00
        # 2. A_01 B_10 -> C_00
        # 3. A_10 B_00 -> C_10
        # 4. A_11 B_10 -> C_10
        
        terms = [
            {'u': {(0,0): 1}, 'v': {(0,0): 1}, 'w': {(0,0): 1}},
            {'u': {(0,1): 1}, 'v': {(1,0): 1}, 'w': {(0,0): 1}},
            {'u': {(1,0): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1}},
            {'u': {(1,1): 1}, 'v': {(1,0): 1}, 'w': {(1,0): 1}},
        ]
        
        for term in terms:
            u, v, w = init_vecs()
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    # Block 1: A_L (j=0,1) with B_T (j=0,1)
    # B_T split into k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_trivial_2x2x1(0, 0, 4)
    
    # Block 2: A_R (j=2,3) with B_B (j=2,3)
    # B_B split into k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_trivial_2x2x1(2, 2, 4)
    
    # Convert to lists of lists for return
    # U is currently list of columns. Need list of rows.
    rank = len(U)
    U_out = [[0.0]*rank for _ in range(n*m)]
    V_out = [[0.0]*rank for _ in range(m*p)]
    W_out = [[0.0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_out[i][r] = U[r][i]
        for j in range(m*p):
            V_out[j][r] = V[r][j]
        for k in range(p*n):
            W_out[k][r] = W[r][k]
            
    return {"U": U_out, "V": V_out, "W": W_out}
