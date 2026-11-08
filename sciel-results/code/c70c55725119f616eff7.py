def construct():
    n, m, p = 3, 3, 7
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def add_strassen_block(u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11, c):
        # M1
        U[u00][c] += 1; U[u11][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w11][c] += 1
        c += 1
        # M2
        U[u10][c] += 1; U[u11][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1; W[w11][c] -= 1
        c += 1
        # M3
        U[u00][c] += 1
        V[v01][c] += 1; V[v11][c] -= 1
        W[w01][c] += 1
        c += 1
        # M4
        U[u11][c] += 1
        V[v10][c] += 1; V[v00][c] -= 1
        W[w00][c] -= 1; W[w10][c] += 1
        c += 1
        # M5
        U[u00][c] += 1; U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] += 1
        c += 1
        # M6
        U[u10][c] += 1; U[u00][c] -= 1
        V[v00][c] += 1; V[v01][c] += 1
        W[w11][c] += 1
        c += 1
        # M7
        U[u01][c] += 1; U[u11][c] -= 1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] -= 1; W[w01][c] += 1
        c += 1
        return c

    # Block 1a: A22 * B22a (k=0,1)
    col = add_strassen_block(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1),
        col
    )
    
    # Block 1b: A22 * B22b (k=2,3)
    col = add_strassen_block(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
        w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1),
        col
    )
    
    # Block 1c: A21 * B14
    # A21: (0,2), (1,2). B14: (2,0), (2,1), (2,2), (2,3)
    for k in range(4):
        U[u_idx(0,2)][col] += 1
        V[v_idx(2,k)][col] += 1
        W[w_idx(k,0)][col] += 1
        col += 1
        U[u_idx(1,2)][col] += 1
        V[v_idx(2,k)][col] += 1
        W[w_idx(k,1)][col] += 1
        col += 1
        
    # Block 2a: A22 * B22c (k=4,5)
    col = add_strassen_block(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,4), v_idx(0,5), v_idx(1,4), v_idx(1,5),
        w_idx(4,0), w_idx(4,1), w_idx(5,0), w_idx(5,1),
        col
    )
    
    # Block 2b: A22 * B21c (k=6)
    # A22: (0,0),(0,1),(1,0),(1,1). B21c: (0,6),(1,6)
    # C60 = A00*B06 + A01*B16
    # C61 = A10*B06 + A11*B16
    U[u_idx(0,0)][col] += 1; V[v_idx(0,6)][col] += 1; W[w_idx(6,0)][col] += 1; col += 1
    U[u_idx(0,1)][col] += 1; V[v_idx(1,6)][col] += 1; W[w_idx(6,0)][col] += 1; col += 1
    U[u_idx(1,0)][col] += 1; V[v_idx(0,6)][col] += 1; W[w_idx(6,1)][col] += 1; col += 1
    U[u_idx(1,1)][col] += 1; V[v_idx(1,6)][col] += 1; W[w_idx(6,1)][col] += 1; col += 1
    
    # Block 2c: A21 * B13
    # A21: (0,2),(1,2). B13: (2,4),(2,5),(2,6)
    for k in [4,5,6]:
        U[u_idx(0,2)][col] += 1
        V[v_idx(2,k)][col] += 1
        W[w_idx(k,0)][col] += 1
        col += 1
        U[u_idx(1,2)][col] += 1
        V[v_idx(2,k)][col] += 1
        W[w_idx(k,1)][col] += 1
        col += 1
        
    # Block 3a: A12 * B24
    # A12: (2,0),(2,1). B24: (0,0..3),(1,0..3)
    for k in range(4):
        U[u_idx(2,0)][col] += 1; V[v_idx(0,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        U[u_idx(2,1)][col] += 1; V[v_idx(1,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        
    # Block 3b: A11 * B14
    # A11: (2,2). B14: (2,0..3)
    for k in range(4):
        U[u_idx(2,2)][col] += 1; V[v_idx(2,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        
    # Block 4a: A12 * B23
    # A12: (2,0),(2,1). B23: (0,4..6),(1,4..6)
    for k in [4,5,6]:
        U[u_idx(2,0)][col] += 1; V[v_idx(0,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        U[u_idx(2,1)][col] += 1; V[v_idx(1,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        
    # Block 4b: A11 * B13
    # A11: (2,2). B13: (2,4..6)
    for k in [4,5,6]:
        U[u_idx(2,2)][col] += 1; V[v_idx(2,k)][col] += 1; W[w_idx(k,2)][col] += 1; col += 1
        
    return {"U": U, "V": V, "W": W}
