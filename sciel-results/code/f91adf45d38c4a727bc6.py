import time

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS

def construct():
    # Total rank: 3 (Strassen blocks) * 7 + 12 (Outer product) = 33
    rank = 33
    
    # Initialize matrices
    # U shape: (n*m, rank) = (6, 33)
    # V shape: (m*p, rank) = (18, 33)
    # W shape: (p*n, rank) = (12, 33)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A_L B_L
    # A_L is 2x2 (cols 0,1 of A)
    # B_L is 2x6 (rows 0,1 of B)
    # Split B_L into 3 blocks of 2x2
    for b in range(3):
        # Block b corresponds to columns 2b, 2b+1 of B_L
        # Inputs from A:
        # a = A[0,0], b_ = A[0,1], c = A[1,0], d = A[1,1]
        # Indices in U:
        u_a = 0 * m + 0  # 0
        u_b = 0 * m + 1  # 1
        u_c = 1 * m + 0  # 3
        u_d = 1 * m + 1  # 4
        
        # Inputs from B:
        # e = B[0, 2b], f = B[0, 2b+1], g = B[1, 2b], h = B[1, 2b+1]
        # Indices in V:
        v_e = 0 * p + 2 * b
        v_f = 0 * p + 2 * b + 1
        v_g = 1 * p + 2 * b
        v_h = 1 * p + 2 * b + 1
        
        # Outputs to C:
        # C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
        # Indices in W:
        w_00 = 2 * b * n + 0
        w_01 = 2 * b * n + 1
        w_10 = (2 * b + 1) * n + 0
        w_11 = (2 * b + 1) * n + 1
        
        # Strassen Terms
        # m1 = (a+d)(e+h)
        U[u_a][r] += 1; U[u_d][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_00][r] += 1; W[w_11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[u_c][r] += 1; U[u_d][r] += 1
        V[v_e][r] += 1
        W[w_01][r] += 1; W[w_11][r] += 1
        r += 1
        
        # m3 = a(f-h)
        U[u_a][r] += 1
        V[v_f][r] += 1; V[v_h][r] -= 1
        W[w_01][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[u_d][r] += 1
        V[v_g][r] += 1; V[v_e][r] -= 1
        W[w_10][r] += 1; W[w_11][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[u_a][r] += 1; U[u_b][r] += 1
        V[v_h][r] += 1
        W[w_00][r] -= 1; W[w_10][r] += 1
        r += 1
        
        # m6 = (c-d)(f+h)
        U[u_c][r] += 1; U[u_d][r] -= 1
        V[v_f][r] += 1; V[v_h][r] += 1
        W[w_11][r] -= 1
        r += 1
        
        # m7 = (a-c)(e+g)
        U[u_a][r] += 1; U[u_c][r] -= 1
        V[v_e][r] += 1; V[v_g][r] += 1
        W[w_00][r] -= 1; W[w_10][r] -= 1
        r += 1
        
    # Part 2: A_R B_R
    # A_R is col 2 of A (2x1)
    # B_R is row 2 of B (1x6)
    # C[i, k] = A[i, 2] * B[2, k]
    # 12 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    return {"U": U, "V": V, "W": W}
