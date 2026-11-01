"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    U_rows = n * m      # 6
    V_rows = m * p      # 30
    W_rows = p * n      # 20
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Block 1: first 2 columns of B (k=0,1)
    # A1 (2x2) * B11 (2x2) using Strassen: 7 terms
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # where a_ij = A[i,j], b_jk = B[j,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1 = (a00+a11)*(b00+b11)
    u = [0]*U_rows; u[0] = 1; u[4] = 1
    v = [0]*V_rows; v[0] = 1; v[11] = 1
    w = [0]*W_rows; w[0] = 1; w[3] = 1
    add_term(u, v, w)
    
    # M2 = (a10+a11)*b00
    u = [0]*U_rows; u[3] = 1; u[4] = 1
    v = [0]*V_rows; v[0] = 1
    w = [0]*W_rows; w[2] = 1
    add_term(u, v, w)
    
    # M3 = a00*(b01-b11)
    u = [0]*U_rows; u[0] = 1
    v = [0]*V_rows; v[1] = 1; v[11] = -1
    w = [0]*W_rows; w[1] = 1
    add_term(u, v, w)
    
    # M4 = a11*(b10-b00)
    u = [0]*U_rows; u[4] = 1
    v = [0]*V_rows; v[10] = 1; v[0] = -1
    w = [0]*W_rows; w[2] = 1
    add_term(u, v, w)
    
    # M5 = (a00+a01)*b11
    u = [0]*U_rows; u[0] = 1; u[1] = 1
    v = [0]*V_rows; v[11] = 1
    w = [0]*W_rows; w[0] = -1; w[1] = 1
    add_term(u, v, w)
    
    # M6 = (a10-a00)*(b00+b01)
    u = [0]*U_rows; u[3] = 1; u[0] = -1
    v = [0]*V_rows; v[0] = 1; v[1] = 1
    w = [0]*W_rows; w[3] = 1
    add_term(u, v, w)
    
    # M7 = (a01-a11)*(b10+b11)
    u = [0]*U_rows; u[1] = 1; u[4] = -1
    v = [0]*V_rows; v[10] = 1; v[11] = 1
    w = [0]*W_rows; w[0] = 1
    add_term(u, v, w)
    
    # Schoolbook for A[:,2] * B[2,0:2]: 4 terms
    for i in range(n):
        for k in range(2):
            u = [0]*U_rows; u[i*m + 2] = 1
            v = [0]*V_rows; v[2*p + k] = 1
            w = [0]*W_rows; w[k*n + i] = 1
            add_term(u, v, w)
    
    # Block 2-5: remaining 8 columns of B (k=2..9)
    # Use schoolbook: 2*3*8 = 48 terms
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u = [0]*U_rows; u[i*m + j] = 1
                v = [0]*V_rows; v[j*p + k] = 1
                w = [0]*W_rows; w[k*n + i] = 1
                add_term(u, v, w)
    
    rank = len(U_list)
    
    # Transpose: U is (U_rows, rank), V is (V_rows, rank), W is (W_rows, rank)
    U_out = [[U_list[c][r] for c in range(rank)] for r in range(U_rows)]
    V_out = [[V_list[c][r] for c in range(rank)] for r in range(V_rows)]
    W_out = [[W_list[c][r] for c in range(rank)] for r in range(W_rows)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
