"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(list(u_vec))
        V.append(list(v_vec))
        W.append(list(w_vec))
    
    # Strassen for each pair of columns (c0, c1)
    for t in range(5):
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # M1 = (a11+a22)(b11+b22)
        # C[0][c0] += 1, C[1][c1] += 1
        u = [1, 0, 0, 1]
        v = [0]*22
        v[0*11 + c0] = 1
        v[1*11 + c1] = 1
        w = [0]*22
        w[2*c0 + 0] = 1
        w[2*c1 + 1] = 1
        add_term(u, v, w)
        
        # M2 = (a21+a22)b11
        # C[1][c0] += 1
        u = [0, 0, 1, 1]
        v = [0]*22
        v[0*11 + c0] = 1
        w = [0]*22
        w[2*c0 + 1] = 1
        add_term(u, v, w)
        
        # M3 = a11(b12-b22)
        # C[0][c1] += 1, C[1][c1] += 1
        u = [1, 0, 0, 0]
        v = [0]*22
        v[0*11 + c1] = 1
        v[1*11 + c1] = -1
        w = [0]*22
        w[2*c1 + 0] = 1
        w[2*c1 + 1] = 1
        add_term(u, v, w)
        
        # M4 = a22(b21-b11)
        # C[0][c0] += 1, C[1][c0] += 1
        u = [0, 0, 0, 1]
        v = [0]*22
        v[1*11 + c0] = 1
        v[0*11 + c0] = -1
        w = [0]*22
        w[2*c0 + 0] = 1
        w[2*c0 + 1] = 1
        add_term(u, v, w)
        
        # M5 = (a11+a12)b22
        # C[0][c0] += -1, C[1][c1] += 1
        u = [1, 1, 0, 0]
        v = [0]*22
        v[1*11 + c1] = 1
        w = [0]*22
        w[2*c0 + 0] = -1
        w[2*c1 + 1] = 1
        add_term(u, v, w)
        
        # M6 = (a21-a11)(b11+b12)
        # C[1][c1] += 1
        u = [-1, 0, 1, 0]
        v = [0]*22
        v[0*11 + c0] = 1
        v[0*11 + c1] = 1
        w = [0]*22
        w[2*c1 + 1] = 1
        add_term(u, v, w)
        
        # M7 = (a12-a22)(b21+b22)
        # C[0][c0] += 1
        u = [0, 1, 0, -1]
        v = [0]*22
        v[1*11 + c0] = 1
        v[1*11 + c1] = 1
        w = [0]*22
        w[2*c0 + 0] = 1
        add_term(u, v, w)
    
    # Schoolbook for column 10
    c = 10
    
    # a11 * B[0][10] -> C[0][10]
    u = [1, 0, 0, 0]
    v = [0]*22
    v[0*11 + c] = 1
    w = [0]*22
    w[2*c + 0] = 1
    add_term(u, v, w)
    
    # a12 * B[1][10] -> C[0][10]
    u = [0, 1, 0, 0]
    v = [0]*22
    v[1*11 + c] = 1
    w = [0]*22
    w[2*c + 0] = 1
    add_term(u, v, w)
    
    # a21 * B[0][10] -> C[1][10]
    u = [0, 0, 1, 0]
    v = [0]*22
    v[0*11 + c] = 1
    w = [0]*22
    w[2*c + 1] = 1
    add_term(u, v, w)
    
    # a22 * B[1][10] -> C[1][10]
    u = [0, 0, 0, 1]
    v = [0]*22
    v[1*11 + c] = 1
    w = [0]*22
    w[2*c + 1] = 1
    add_term(u, v, w)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
