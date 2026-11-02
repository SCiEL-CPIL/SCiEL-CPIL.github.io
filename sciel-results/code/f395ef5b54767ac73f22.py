"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: 4 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    # Total rank = 4*7 + 4 = 32
    
    # Initialize matrices
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx][col_offset] = u_val
        V[v_idx][col_offset] = v_val
        W[w_idx][col_offset] = w_val

    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Indices:
    # a: (0,0) -> 0
    # b: (0,1) -> 1
    # c: (1,0) -> 2
    # d: (1,1) -> 3
    # e: (0,0) -> 0
    # f: (0,1) -> 1
    # g: (1,0) -> 2
    # h: (1,1) -> 3
    
    # Terms:
    # m1 = (a+b)(e+g)
    # m2 = (a-b)(e-g)
    # m3 = (c+d)(e+g)
    # m4 = (c-d)(e-g)
    # m5 = (a+c)(f+h)
    # m6 = (a-c)(f-h)
    # m7 = (b+d)(f+h)
    
    # Output:
    # C11 = m1 + m4 - m3 - m7
    # C12 = m7 + m5
    # C21 = m2 + m3
    # C22 = m1 - m2 + m4 - m6
    
    def add_strassen_block(b_col_offset, c_col_offset):
        nonlocal col_offset
        # b_col_offset is the column index in B for the first column of the 2x2 block
        # c_col_offset is the column index in C for the first column of the 2x2 block
        
        # m1
        add_term(0, b_col_offset, c_col_offset, 1, 1, 1) # a, e, C11
        add_term(0, b_col_offset+1, c_col_offset, 0, 0, 0) # b, f, C11 (0)
        add_term(1, b_col_offset, c_col_offset, 1, 1, 0) # b, e, C11 (0) -- Wait, U is for A, V for B, W for C
        # U index: A row-major? 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V index: B row-major? 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W index: C row-major? 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # m1 = (a+b)(e+g)
        # U: a(0)=1, b(1)=1
        # V: e(0)=1, g(2)=1
        # W: C11(0)=1
        U[0][col_offset] = 1
        U[1][col_offset] = 1
        V[b_col_offset][col_offset] = 1
        V[b_col_offset+2][col_offset] = 1
        W[c_col_offset][col_offset] = 1
        col_offset += 1
        
        # m2 = (a-b)(e-g)
        U[0][col_offset] = 1
        U[1][col_offset] = -1
        V[b_col_offset][col_offset] = 1
        V[b_col_offset+2][col_offset] = -1
        W[c_col_offset+2][col_offset] = 1 # C21
        col_offset += 1
        
        # m3 = (c+d)(e+g)
        U[2][col_offset] = 1
        U[3][col_offset] = 1
        V[b_col_offset][col_offset] = 1
        V[b_col_offset+2][col_offset] = 1
        W[c_col_offset+2][col_offset] = 1 # C21
        col_offset += 1
        
        # m4 = (c-d)(e-g)
        U[2][col_offset] = 1
        U[3][col_offset] = -1
        V[b_col_offset][col_offset] = 1
        V[b_col_offset+2][col_offset] = -1
        W[c_col_offset][col_offset] = 1 # C11
        col_offset += 1
        
        # m5 = (a+c)(f+h)
        U[0][col_offset] = 1
        U[2][col_offset] = 1
        V[b_col_offset+1][col_offset] = 1
        V[b_col_offset+3][col_offset] = 1
        W[c_col_offset+1][col_offset] = 1 # C12
        col_offset += 1
        
        # m6 = (a-c)(f-h)
        U[0][col_offset] = 1
        U[2][col_offset] = -1
        V[b_col_offset+1][col_offset] = 1
        V[b_col_offset+3][col_offset] = -1
        W[c_col_offset+3][col_offset] = 1 # C22
        col_offset += 1
        
        # m7 = (b+d)(f+h)
        U[1][col_offset] = 1
        U[3][col_offset] = 1
        V[b_col_offset+1][col_offset] = 1
        V[b_col_offset+3][col_offset] = 1
        W[c_col_offset][col_offset] = -1 # C11 (from -m7)
        W[c_col_offset+1][col_offset] = 1 # C12 (from +m7)
        col_offset += 1

    # Schoolbook for 2x2x1
    # A = [[a, b], [c, d]], B = [[e], [g]]
    # C = [[ae+bg], [ce+dg]]
    # 4 terms:
    # 1: a*e -> C11
    # 2: b*g -> C11
    # 3: c*e -> C21
    # 4: d*g -> C21
    
    def add_schoolbook_block(b_col_offset, c_col_offset):
        nonlocal col_offset
        # 1: a*e
        U[0][col_offset] = 1
        V[b_col_offset][col_offset] = 1
        W[c_col_offset][col_offset] = 1
        col_offset += 1
        
        # 2: b*g
        U[1][col_offset] = 1
        V[b_col_offset+2][col_offset] = 1
        W[c_col_offset][col_offset] = 1
        col_offset += 1
        
        # 3: c*e
        U[2][col_offset] = 1
        V[b_col_offset][col_offset] = 1
        W[c_col_offset+2][col_offset] = 1
        col_offset += 1
        
        # 4: d*g
        U[3][col_offset] = 1
        V[b_col_offset+2][col_offset] = 1
        W[c_col_offset+2][col_offset] = 1
        col_offset += 1

    # Apply 4 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7
    for i in range(4):
        b_col = i * 2
        c_col = i * 2
        add_strassen_block(b_col, c_col)
        
    # Apply 1 Schoolbook block for column 8
    add_schoolbook_block(8, 8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
