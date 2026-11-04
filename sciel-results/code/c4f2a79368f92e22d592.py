def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0]*rank for _ in range(n*m)]  # 10 x 46
    V = [[0]*rank for _ in range(m*p)]  # 25 x 46
    W = [[0]*rank for _ in range(p*n)]  # 10 x 46
    
    col = 0
    
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        # M1 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
        U[a_r0*m + a_c0][col] += 1
        U[a_r1*m + a_c1][col] += 1
        V[b_r0*p + b_c0][col] += 1
        V[b_r1*p + b_c1][col] += 1
        W[c_c0*n + c_r0][col] += 1
        W[c_c1*n + c_r1][col] += 1
        col += 1
        # M2 = (a10+a11)*b00 -> c10+=1
        U[a_r1*m + a_c0][col] += 1
        U[a_r1*m + a_c1][col] += 1
        V[b_r0*p + b_c0][col] += 1
        W[c_c0*n + c_r1][col] += 1
        col += 1
        # M3 = a00*(b01-b11) -> c01+=1
        U[a_r0*m + a_c0][col] += 1
        V[b_r0*p + b_c1][col] += 1
        V[b_r1*p + b_c1][col] -= 1
        W[c_c1*n + c_r0][col] += 1
        col += 1
        # M4 = a11*(b10-b00) -> c00+=1, c10+=1
        U[a_r1*m + a_c1][col] += 1
        V[b_r1*p + b_c0][col] += 1
        V[b_r0*p + b_c0][col] -= 1
        W[c_c0*n + c_r0][col] += 1
        W[c_c0*n + c_r1][col] += 1
        col += 1
        # M5 = (a00+a01)*b11 -> c00-=1, c01+=1
        U[a_r0*m + a_c0][col] += 1
        U[a_r0*m + a_c1][col] += 1
        V[b_r1*p + b_c1][col] += 1
        W[c_c0*n + c_r0][col] -= 1
        W[c_c1*n + c_r0][col] += 1
        col += 1
        # M6 = (a10-a00)*(b00+b01) -> c11+=1
        U[a_r1*m + a_c0][col] += 1
        U[a_r0*m + a_c0][col] -= 1
        V[b_r0*p + b_c0][col] += 1
        V[b_r0*p + b_c1][col] += 1
        W[c_c1*n + c_r1][col] += 1
        col += 1
        # M7 = (a01-a11)*(b10+b11) -> c00+=1
        U[a_r0*m + a_c1][col] += 1
        U[a_r1*m + a_c1][col] -= 1
        V[b_r1*p + b_c0][col] += 1
        V[b_r1*p + b_c1][col] += 1
        W[c_c0*n + c_r0][col] += 1
        col += 1

    def add_outer(a_r0, a_r1, a_c, b_r, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        U[a_r0*m + a_c][col] += 1
        V[b_r*p + b_c0][col] += 1
        W[c_c0*n + c_r0][col] += 1
        col += 1
        U[a_r0*m + a_c][col] += 1
        V[b_r*p + b_c1][col] += 1
        W[c_c1*n + c_r0][col] += 1
        col += 1
        U[a_r1*m + a_c][col] += 1
        V[b_r*p + b_c0][col] += 1
        W[c_c0*n + c_r1][col] += 1
        col += 1
        U[a_r1*m + a_c][col] += 1
        V[b_r*p + b_c1][col] += 1
        W[c_c1*n + c_r1][col] += 1
        col += 1

    def add_2x2_2x1(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c, c_r0, c_r1, c_c):
        nonlocal col
        U[a_r0*m + a_c0][col] += 1
        V[b_r0*p + b_c][col] += 1
        W[c_c*n + c_r0][col] += 1
        col += 1
        U[a_r0*m + a_c1][col] += 1
        V[b_r1*p + b_c][col] += 1
        W[c_c*n + c_r0][col] += 1
        col += 1
        U[a_r1*m + a_c0][col] += 1
        V[b_r0*p + b_c][col] += 1
        W[c_c*n + c_r1][col] += 1
        col += 1
        U[a_r1*m + a_c1][col] += 1
        V[b_r1*p + b_c][col] += 1
        W[c_c*n + c_r1][col] += 1
        col += 1

    def add_2x1_1x1(a_r0, a_r1, a_c, b_r, b_c, c_r0, c_r1, c_c):
        nonlocal col
        U[a_r0*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[c_c*n + c_r0][col] += 1
        col += 1
        U[a_r1*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[c_c*n + c_r1][col] += 1
        col += 1

    # C[*,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2] + A[:,4]*B[4,0:2]
    add_strassen(0,1, 0,1,  0,1, 0,1,  0,1, 0,1)
    add_strassen(0,1, 2,3,  2,3, 0,1,  0,1, 0,1)
    add_outer(0,1, 4,  4, 0,1,  0,1, 0,1)
    
    # C[*,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4] + A[:,4]*B[4,2:4]
    add_strassen(0,1, 0,1,  0,1, 2,3,  0,1, 2,3)
    add_strassen(0,1, 2,3,  2,3, 2,3,  0,1, 2,3)
    add_outer(0,1, 4,  4, 2,3,  0,1, 2,3)
    
    # C[*,4] = A[:,0:2]*B[0:2,4] + A[:,2:4]*B[2:4,4] + A[:,4]*B[4,4]
    add_2x2_2x1(0,1, 0,1,  0,1, 4,  0,1, 4)
    add_2x2_2x1(0,1, 2,3,  2,3, 4,  0,1, 4)
    add_2x1_1x1(0,1, 4,  4, 4,  0,1, 4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
