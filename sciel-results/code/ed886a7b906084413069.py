"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 43 construction:
    # 7 terms for Strassen (first 2 columns)
    # 4 * 9 = 36 terms for standard (remaining 9 columns)
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for k=0, 1
    # m1 = (a+d)(x0+x1) ... wait, Strassen is for 2x2x2.
    # B is 2x11. We apply Strassen to B[:, 0:2].
    # A is 2x2.
    # C[:, 0:2] = A * B[:, 0:2].
    
    # Strassen intermediates:
    # 1: (a+d) * (x0+x1)
    # 2: (c+d) * x0
    # 3: a * (x1-x0)
    # 4: d * (x1-x0) ? No, standard Strassen:
    # m1 = (a+d)(x+w)
    # m2 = (c+d)x
    # m3 = a(y-x)
    # m4 = d(w-z)
    # m5 = (a+b)w
    # m6 = (c-a)(x+z)
    # m7 = (b-d)(y+w)
    
    # Let's map indices for B[:, 0:2]
    # B[0,0] idx 0, B[0,1] idx 1
    # B[1,0] idx 2, B[1,1] idx 3
    
    # Term 1: m1
    U[0][col] = 1; U[2][col] = 1 # a+d
    V[0][col] = 1; V[3][col] = 1 # x0 + x1 (B[0,0] + B[1,1])
    W[0][col] = 1; W[3][col] = 1 # C[0,0] + C[1,1]
    col += 1
    
    # Term 2: m2
    U[2][col] = 1; U[3][col] = 1 # c+d
    V[0][col] = 1 # x0
    W[2][col] = 1 # C[1,0]
    col += 1
    
    # Term 3: m3
    U[0][col] = 1 # a
    V[1][col] = 1; V[0][col] = -1 # y-x (B[0,1] - B[0,0])
    W[1][col] = 1; W[2][col] = 1 # C[0,1] + C[1,0]
    col += 1
    
    # Term 4: m4
    U[3][col] = 1 # d
    V[3][col] = 1; V[2][col] = -1 # w-z (B[1,1] - B[1,0])
    W[0][col] = 1; W[2][col] = 1 # C[0,0] + C[1,0]
    col += 1
    
    # Term 5: m5
    U[0][col] = 1; U[1][col] = 1 # a+b
    V[3][col] = 1 # w
    W[0][col] = -1; W[1][col] = 1 # -C[0,0] + C[0,1]
    col += 1
    
    # Term 6: m6
    U[2][col] = 1; U[0][col] = -1 # c-a
    V[0][col] = 1; V[2][col] = 1 # x+z
    W[3][col] = 1 # C[1,1]
    col += 1
    
    # Term 7: m7
    U[1][col] = 1; U[3][col] = -1 # b-d
    V[1][col] = 1; V[3][col] = 1 # y+w
    W[0][col] = 1; W[1][col] = -1 # C[0,0] - C[0,1]
    col += 1
    
    # Standard terms for k=2 to 10
    for k in range(2, p):
        # a * B[0,k]
        U[0][col] = 1
        V[k][col] = 1 # B[0,k] index is 0*p + k = k
        W[k*n][col] = 1 # C[0,k] index is k*n + 0
        col += 1
        
        # b * B[1,k]
        U[1][col] = 1
        V[p+k][col] = 1 # B[1,k] index is 1*p + k
        W[k*n][col] = 1
        col += 1
        
        # c * B[0,k]
        U[2][col] = 1
        V[k][col] = 1
        W[k*n+1][col] = 1 # C[1,k] index is k*n + 1
        col += 1
        
        # d * B[1,k]
        U[3][col] = 1
        V[p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
