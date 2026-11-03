"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition:
    # 12 from A[:,0] * B[0,:]
    # 21 from A[:,1:3] * B[1:3,:] using 3 Strassen 2x2x2 blocks
    
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: A[:, 0] * B[0, :]
    # A indices: (i, 0) -> i*m + 0
    # B indices: (0, k) -> 0*p + k
    # C indices: (i, k) -> k*n + i
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # A is 2x2, B is 2x6.
    # Split B into 3 blocks of 2x2.
    # Each block uses Strassen (rank 7).
    
    # Strassen factors definition
    # M1 = (b-d)(e+f)
    # M2 = (a+b)h
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+d)(e+h)
    # M6 = (c-d)(g+h)
    # M7 = (b+c)e
    
    # A elements (global indices):
    # a = A[0,1] -> 0*m + 1
    # b = A[0,2] -> 0*m + 2
    # c = A[1,1] -> 1*m + 1
    # d = A[1,2] -> 1*m + 2
    
    idx_a = 0 * m + 1
    idx_b = 0 * m + 2
    idx_c = 1 * m + 1
    idx_d = 1 * m + 2
    
    for t in range(3):
        # Block t uses B columns 2t, 2t+1
        # B rows 1, 2
        # e = B[1, 2t] -> 1*p + 2t
        # f = B[1, 2t+1] -> 1*p + 2t+1
        # g = B[2, 2t] -> 2*p + 2t
        # h = B[2, 2t+1] -> 2*p + 2t+1
        
        idx_e = 1 * p + 2 * t
        idx_f = 1 * p + 2 * t + 1
        idx_g = 2 * p + 2 * t
        idx_h = 2 * p + 2 * t + 1
        
        # Output C columns 2t, 2t+1
        # C rows 0, 1
        
        # M1
        r = col
        U[idx_b][r] += 1.0
        U[idx_d][r] -= 1.0
        V[idx_e][r] += 1.0
        V[idx_f][r] += 1.0
        W[(2 * t) * n + 0][r] += 1.0      # C[0, 2t]
        W[(2 * t + 1) * n + 0][r] += 1.0    # C[0, 2t+1]
        W[(2 * t + 1) * n + 1][r] -= 1.0    # C[1, 2t+1]
        col += 1
        
        # M2
        r = col
        U[idx_a][r] += 1.0
        U[idx_b][r] += 1.0
        V[idx_h][r] += 1.0
        W[(2 * t) * n + 0][r] -= 1.0      # C[0, 2t]
        W[(2 * t + 1) * n + 0][r] += 1.0    # C[0, 2t+1]
        col += 1
        
        # M3
        r = col
        U[idx_a][r] += 1.0
        V[idx_f][r] += 1.0
        V[idx_h][r] -= 1.0
        W[(2 * t) * n + 1][r] += 1.0      # C[1, 2t]
        W[(2 * t + 1) * n + 1][r] += 1.0    # C[1, 2t+1]
        col += 1
        
        # M4
        r = col
        U[idx_d][r] += 1.0
        V[idx_e][r] += 1.0
        V[idx_g][r] -= 1.0
        W[(2 * t) * n + 0][r] += 1.0      # C[0, 2t]
        W[(2 * t) * n + 1][r] += 1.0      # C[1, 2t]
        col += 1
        
        # M5
        r = col
        U[idx_a][r] += 1.0
        U[idx_d][r] += 1.0
        V[idx_e][r] += 1.0
        V[idx_h][r] += 1.0
        W[(2 * t) * n + 0][r] += 1.0      # C[0, 2t]
        W[(2 * t + 1) * n + 1][r] += 1.0    # C[1, 2t+1]
        col += 1
        
        # M6
        r = col
        U[idx_c][r] += 1.0
        U[idx_d][r] -= 1.0
        V[idx_g][r] += 1.0
        V[idx_h][r] += 1.0
        W[(2 * t) * n + 0][r] += 1.0      # C[0, 2t]
        col += 1
        
        # M7
        r = col
        U[idx_b][r] += 1.0
        U[idx_c][r] += 1.0
        V[idx_e][r] += 1.0
        W[(2 * t + 1) * n + 1][r] += 1.0    # C[1, 2t+1]
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
