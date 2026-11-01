def add_block(i_off, j_off, k_off, col_off):
    # Local to Global
    def U_idx(li, lj):
        return (i_off + li) * m + (j_off + lj)
    def V_idx(lj, lk):
        return (j_off + lj) * p + (k_off + lk)
    def W_idx(lk, li):
        return (k_off + lk) * n + (i_off + li)

    # M1 = (a+d)(e+h)
    # a: (0,0), d: (1,1)
    U[U_idx(0,0)][col_off] += 1
    U[U_idx(1,1)][col_off] += 1
    V[V_idx(0,0)][col_off] += 1
    V[V_idx(1,1)][col_off] += 1
    W[W_idx(0,0)][col_off] += 1 # C00
    W[W_idx(1,1)][col_off] += 1 # C11

    # M2 = (c+d)e
    # c: (1,0), d: (1,1), e: (0,0)
    U[U_idx(1,0)][col_off+1] += 1
    U[U_idx(1,1)][col_off+1] += 1
    V[V_idx(0,0)][col_off+1] += 1
    W[W_idx(0,1)][col_off+1] += 1 # C10
    W[W_idx(1,1)][col_off+1] -= 1 # C11

    # M3 = a(f-h)
    # a: (0,0), f: (0,1), h: (1,1)
    U[U_idx(0,0)][col_off+2] += 1
    V[V_idx(0,1)][col_off+2] += 1
    V[V_idx(1,1)][col_off+2] -= 1
    W[W_idx(1,0)][col_off+2] += 1 # C01
    W[W_idx(1,1)][col_off+2] += 1 # C11

    # M4 = d(g-e)
    # d: (1,1), g: (1,0), e: (0,0)
    U[U_idx(1,1)][col_off+3] += 1
    V[V_idx(1,0)][col_off+3] += 1
    V[V_idx(0,0)][col_off+3] -= 1
    W[W_idx(0,0)][col_off+3] += 1 # C00
    W[W_idx(0,1)][col_off+3] += 1 # C10

    # M5 = (a+b)h
    # a: (0,0), b: (0,1), h: (1,1)
    U[U_idx(0,0)][col_off+4] += 1
    U[U_idx(0,1)][col_off+4] += 1
    V[V_idx(1,1)][col_off+4] += 1
    W[W_idx(0,0)][col_off+4] -= 1 # C00
    W[W_idx(1,0)][col_off+4] += 1 # C01

    # M6 = (c-a)(e+f)
    # c: (1,0), a: (0,0), e: (0,0), f: (0,1)
    U[U_idx(1,0)][col_off+5] += 1
    U[U_idx(0,0)][col_off+5] -= 1
    V[V_idx(0,0)][col_off+5] += 1
    V[V_idx(0,1)][col_off+5] += 1
    W[W_idx(1,1)][col_off+5] += 1 # C11

    # M7 = (b-d)(g+h)
    # b: (0,1), d: (1,1), g: (1,0), h: (1,1)
    U[U_idx(0,1)][col_off+6] += 1
    U[U_idx(1,1)][col_off+6] -= 1
    V[V_idx(1,0)][col_off+6] += 1
    V[V_idx(1,1)][col_off+6] += 1
    W[W_idx(0,0)][col_off+6] += 1 # C00
    W[W_idx(1,1)][col_off+6] += 1 # C11
