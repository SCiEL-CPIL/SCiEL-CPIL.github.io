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
        
        # Indices in U for A_L (2x2)
        # A is 2x3, so A[0,0] is index 0, A[0,1] is 1, A[1,0] is 3, A[1,1] is 4
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # Indices in V for B_L block b (2x2)
        # B is 3x6. Rows 0,1. Cols 2b, 2b+1.
        # V index = row * p + col
        v_b00 = 0 * p + 2 * b
        v_b01 = 0 * p + 2 * b + 1
        v_b10 = 1 * p + 2 * b
        v_b11 = 1 * p + 2 * b + 1
        
        # Indices in W for C block b (2x2)
        # C is 2x6. W index = col * n + row
        w_c00 = 2 * b * n + 0
        w_c01 = 2 * b * n + 1
        w_c10 = (2 * b + 1) * n + 0
        w_c11 = (2 * b + 1) * n + 1
        
        # Strassen Terms
        # M1 = (A00+A11)(B00+B11)
        U[u_a00][r] += 1; U[u_a11][r] += 1
        V[v_b00][r] += 1; V[v_b11][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # M2 = (A10+A11)B00
        U[u_a10][r] += 1; U[u_a11][r] += 1
        V[v_b00][r] += 1
        W[w_c10][r] += 1
        r += 1
        
        # M3 = A00(B01-B11)
        U[u_a00][r] += 1
        V[v_b01][r] += 1; V[v_b11][r] -= 1
        W[w_c01][r] += 1
        r += 1
        
        # M4 = A11(B10-B00)
        U[u_a11][r] += 1
        V[v_b10][r] += 1; V[v_b00][r] -= 1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # M5 = (A00+A01)B11
        U[u_a00][r] += 1; U[u_a01][r] += 1
        V[v_b11][r] += 1
        W[w_c00][r] -= 1; W[w_c01][r] += 1
        r += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_a10][r] += 1; U[u_a00][r] -= 1
        V[v_b00][r] += 1; V[v_b01][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # M7 = (A01-A10)(B10+B11)
        U[u_a01][r] += 1; U[u_a10][r] -= 1
        V[v_b10][r] += 1; V[v_b11][r] += 1
        W[w_c00][r] += 1; W[w_c10][r] -= 1
        r += 1

    # Part 2: A_R B_R
    # A_R is col 2 of A (2x1)
    # B_R is row 2 of B (1x6)
    # C[i, k] += A[i, 2] * B[2, k]
    # 12 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    return {"U": U, "V": V, "W": W}
