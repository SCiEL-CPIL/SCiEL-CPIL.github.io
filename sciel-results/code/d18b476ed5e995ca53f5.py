"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = [[] for _ in range(n*m)]
    V = [[] for _ in range(m*p)]
    W = [[] for _ in range(p*n)]
    
    def add_term(i, j, k, u_val, v_val, w_val):
        U[i*m + j].append(u_val)
        V[j*p + k].append(v_val)
        W[k*n + i].append(w_val)
        
    def add_2x2x2_strassen(i_off, j_off, k_off):
        # i_off, j_off, k_off are top-left indices.
        # a00: i_off, j_off
        # a01: i_off, j_off+1
        # a10: i_off+1, j_off
        # a11: i_off+1, j_off+1
        # b00: j_off, k_off
        # b01: j_off, k_off+1
        # b10: j_off+1, k_off
        # b11: j_off+1, k_off+1
        # c00: i_off, k_off
        # c01: i_off, k_off+1
        # c10: i_off+1, k_off
        # c11: i_off+1, k_off+1
        
        # m1 = (a00+a11)(b00+b11) -> c00, c11
        add_term(i_off, j_off, k_off, 1, 1, 1) # c00 part of m1? No.
        # This is wrong. add_term adds a single rank-1 term.
        # m1 is one rank-1 term.
        # U: a00, a11. V: b00, b11. W: c00, c11.
        # But add_term takes single indices i,j,k.
        # I need to add multiple entries to U, V, W for a single rank.
        
        pass

    # I need a different way to add terms.
    # U, V, W are lists of coefficients for each rank.
    # So for each rank r, I define U[:, r], V[:, r], W[:, r].
    
    # I will build U, V, W as matrices and then transpose.
    
    U_mat = [[0] * 92 for _ in range(n*m)]
    V_mat = [[0] * 92 for _ in range(m*p)]
    W_mat = [[0] * 92 for _ in range(p*n)]
    
    rank = 0
    
    def add_rank(u_dict, v_dict, w_dict):
        nonlocal rank
        for i, j in u_dict:
            U_mat[i*m + j][rank] = u_dict[(i, j)]
        for j, k in v_dict:
            V_mat[j*p + k][rank] = v_dict[(j, k)]
        for k, i in w_dict:
            W_mat[k*n + i][rank] = w_dict[(k, i)]
        rank += 1
        
    # 2x2x2 Strassen
    def add_strassen(i_off, j_off, k_off):
        # m1
        u = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w = {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        add_rank(u, v, w)
        
        # m2 = (a10+a11)b00 -> c10, c11
        u = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off, k_off): 1}
        w = {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1}
        add_rank(u, v, w)
        
        # m3 = a00(b01-b11) -> c01, c11
        u = {(i_off, j_off): 1}
        v = {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}
        w = {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
        add_rank(u, v, w)
        
        # m4 = a11(b10-b00) -> c00, c10
        u = {(i_off+1, j_off+1): 1}
        v = {(j_off+1, k_off): 1, (j_off, k_off): -1}
        w = {(k_off, i_off): 1, (k_off, i_off+1): 1}
        add_rank(u, v, w)
        
        # m5 = (a00+a01)b11 -> c00, c01
        u = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v = {(j_off+1, k_off+1): 1}
        w = {(k_off, i_off): -1, (k_off+1, i_off): 1}
        add_rank(u, v, w)
        
        # m6 = (a10-a00)(b00+b01) -> c11
        u = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        w = {(k_off+1, i_off+1): 1}
        add_rank(u, v, w)
        
        # m7 = (a01-a11)(b10+b11) -> c00
        u = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        w = {(k_off, i_off): 1}
        add_rank(u, v, w)
        
    # 2x2x1 schoolbook
    def add_2x2x1(i_off, j_off, k_off):
        # a00*b00 -> c00
        add_rank({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(k_off, i_off): 1})
        # a01*b00 -> c00? No.
        # a00*b00, a01*b10? No.
        # B is 2x1. b00, b10.
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        
        # Term 1: a00*b00 -> c00
        # Term 2: a01*b10 -> c00
        # Term 3: a10*b00 -> c10
        # Term 4: a11*b10 -> c10
        
        # We can combine?
        # No, just 4 terms?
        # Rank 2 for 2x2x1?
        # (a00+a01)(b00+b10) = a00b00 + a00b10 + a01b00 + a01b10.
        # We need a00b00 + a01b10.
        # This is not rank 1.
        # Rank 2 is possible?
        # (a00)(b00) + (a01)(b10)
        # (a10)(b00) + (a11)(b10)
        # Total 4 terms.
        
        add_rank({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(k_off, i_off): 1})
        add_rank({(i_off, j_off+1): 1}, {(j_off+1, k_off): 1}, {(k_off, i_off): 1})
        add_rank({(i_off+1, j_off): 1}, {(j_off, k_off): 1}, {(k_off, i_off+1): 1})
        add_rank({(i_off+1, j_off+1): 1}, {(j_off+1, k_off): 1}, {(k_off, i_off+1): 1})

    # 4 blocks of 2x2x7
    # Block 1: i 0-1, j 0-1
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_2x2x1(0, 0, 6)
    
    # Block 2: i 0-1, j 2-3
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    add_2x2x1(0, 2, 6)
    
    # Block 3: i 2-3, j 0-1
    add_strassen(2, 0, 0)
    add_strassen(2, 0, 2)
    add_strassen(2, 0, 4)
    add_2x2x1(2, 0, 6)
    
    # Block 4: i 2-3, j 2-3
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_strassen(2, 2, 4)
    add_2x2x1(2, 2, 6)
    
    U = [list(row) for row in U_mat]
    V = [list(row) for row in V_mat]
    W = [list(row) for row in W_mat]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
