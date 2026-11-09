import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    # Part 1: C2 = A2 * B2 (Rank 10)
    # A2 is col 2 of A (indices 2, 5)
    # B2 is row 2 of B (indices 10, 11, 12, 13, 14)
    # C2 is all of C (indices 0..9)
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(5):
            u_vec = [0] * (n*m)
            u_vec[i * m + 2] = 1
            
            v_vec = [0] * (m*p)
            v_vec[2 * p + k] = 1
            
            w_vec = [0] * (p*n)
            w_vec[k * n + i] = 1
            
            add_term(u_vec, v_vec, w_vec)

    # Part 2a: C1a = A1 * B1a (Rank 7, Strassen)
    # A1 is cols 0,1 of A (indices 0,1,3,4)
    # B1a is cols 0,1 of B (indices 0,1,5,6)
    # C1a is cols 0,1 of C (indices 0,1,2,3)
    
    # Strassen terms for A1, B1a
    # P1 = (a+d)(e+h)
    u = [0]*6; u[0]=1; u[4]=1
    v = [0]*15; v[0]=1; v[6]=1
    w = [0]*10; w[0]=1; w[3]=1
    add_term(u, v, w)
    
    # P2 = (c+d)e
    u = [0]*6; u[3]=1; u[4]=1
    v = [0]*15; v[0]=1
    w = [0]*10; w[2]=1; w[3]=-1
    add_term(u, v, w)
    
    # P3 = a(f-h)
    u = [0]*6; u[0]=1
    v = [0]*15; v[1]=1; v[6]=-1
    w = [0]*10; w[1]=1; w[3]=1
    add_term(u, v, w)
    
    # P4 = d(g-e)
    u = [0]*6; u[4]=1
    v = [0]*15; v[5]=1; v[0]=-1
    w = [0]*10; w[0]=1; w[2]=1
    add_term(u, v, w)
    
    # P5 = (a+b)h
    u = [0]*6; u[0]=1; u[1]=1
    v = [0]*15; v[6]=1
    w = [0]*10; w[0]=-1; w[1]=1
    add_term(u, v, w)
    
    # P6 = (c-a)(e+f)
    u = [0]*6; u[3]=1; u[0]=-1
    v = [0]*15; v[0]=1; v[1]=1
    w = [0]*10; w[3]=1
    add_term(u, v, w)
    
    # P7 = (b-d)(g+h)
    u = [0]*6; u[1]=1; u[4]=-1
    v = [0]*15; v[5]=1; v[6]=1
    w = [0]*10; w[0]=1
    add_term(u, v, w)

    # Part 2b-i: C1b,a = A1 * B1b,a (Rank 7, Strassen)
    # A1 is same
    # B1b,a is cols 2,3 of B (indices 2,3,7,8)
    # C1b,a is cols 2,3 of C (indices 4,5,6,7)
    
    # P1' = (a+d)(e'+h')
    u = [0]*6; u[0]=1; u[4]=1
    v = [0]*15; v[2]=1; v[8]=1
    w = [0]*10; w[4]=1; w[7]=1
    add_term(u, v, w)
    
    # P2' = (c+d)e'
    u = [0]*6; u[3]=1; u[4]=1
    v = [0]*15; v[2]=1
    w = [0]*10; w[6]=1; w[7]=-1
    add_term(u, v, w)
    
    # P3' = a(f'-h')
    u = [0]*6; u[0]=1
    v = [0]*15; v[3]=1; v[8]=-1
    w = [0]*10; w[5]=1; w[7]=1
    add_term(u, v, w)
    
    # P4' = d(g'-e')
    u = [0]*6; u[4]=1
    v = [0]*15; v[7]=1; v[2]=-1
    w = [0]*10; w[4]=1; w[6]=1
    add_term(u, v, w)
    
    # P5' = (a+b)h'
    u = [0]*6; u[0]=1; u[1]=1
    v = [0]*15; v[8]=1
    w = [0]*10; w[4]=-1; w[5]=1
    add_term(u, v, w)
    
    # P6' = (c-a)(e'+f')
    u = [0]*6; u[3]=1; u[0]=-1
    v = [0]*15; v[2]=1; v[3]=1
    w = [0]*10; w[7]=1
    add_term(u, v, w)
    
    # P7' = (b-d)(g'+h')
    u = [0]*6; u[1]=1; u[4]=-1
    v = [0]*15; v[7]=1; v[8]=1
    w = [0]*10; w[4]=1
    add_term(u, v, w)

    # Part 2b-ii: C1b,b = A1 * B1b,b (Rank 4, Schoolbook)
    # A1 is same
    # B1b,b is col 4 of B (indices 4, 9)
    # C1b,b is col 4 of C (indices 8, 9)
    
    # Term 1: A[0,0] * B[0,4] -> C[0,4]
    u = [0]*6; u[0]=1
    v = [0]*15; v[4]=1
    w = [0]*10; w[8]=1
    add_term(u, v, w)
    
    # Term 2: A[0,1] * B[1,4] -> C[0,4]
    u = [0]*6; u[1]=1
    v = [0]*15; v[9]=1
    w = [0]*10; w[8]=1
    add_term(u, v, w)
    
    # Term 3: A[1,0] * B[0,4] -> C[1,4]
    u = [0]*6; u[3]=1
    v = [0]*15; v[4]=1
    w = [0]*10; w[9]=1
    add_term(u, v, w)
    
    # Term 4: A[1,1] * B[1,4] -> C[1,4]
    u = [0]*6; u[4]=1
    v = [0]*15; v[9]=1
    w = [0]*10; w[9]=1
    add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
