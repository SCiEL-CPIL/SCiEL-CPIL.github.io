"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition:
    # Split p=5 into p1=2, p2=2, p3=1.
    # M_2x2x2 (rank 7) + M_2x2x2 (rank 7) + M_2x2x1 (rank 4)
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # m1 = (a+d)(x+w)
    # m2 = (c+d)x
    # m3 = a(y-w)
    # m4 = d(z-x)
    # m5 = (a+b)w
    # m6 = (c-a)(x+z)
    # m7 = (b-d)(y+w)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We need to map these to U, V, W.
    # U indices: i*m + j (4 rows)
    # V indices: j*p + k (10 rows)
    # W indices: k*n + i (10 rows)
    
    col = 0
    def add_strassen_block(offset_k):
        nonlocal col
        # 7 multiplications
        # m1
        U[0][col] = 1 # a (i=0,j=0)
        U[1][col] = 1 # d (i=1,j=1) -> Wait, U is for A.
        # A indices: i in 0..1, j in 0..1.
        # U[i*m+j, r].
        # m1 uses a (0,0) and d (1,1).
        U[0][col] = 1
        U[3][col] = 1
        # V uses x (0,0) and w (1,1) of B block.
        # B block starts at k=offset_k.
        # B indices: j in 0..1, k in offset_k..offset_k+1.
        # V[j*p+k, r].
        V[0*p + offset_k][col] = 1 # x (j=0, k=0)
        V[1*p + offset_k+1][col] = 1 # w (j=1, k=1)
        # W uses C11 (0,0) and C22 (1,1) of output block.
        # C indices: k in offset_k.., i in 0..1.
        # W[k*n+i, r].
        W[offset_k*n + 0][col] = 1 # C11 (k=0, i=0)
        W[(offset_k+1)*n + 1][col] = 1 # C22 (k=1, i=1)
        col += 1
        
        # m2 = (c+d)x
        U[1][col] = 1 # c (i=1,j=0) -> index 1*2+0 = 2? No.
        # i=1, j=0 -> 1*2+0 = 2.
        # i=1, j=1 -> 1*2+1 = 3.
        # i=0, j=0 -> 0.
        # i=0, j=1 -> 1.
        U[2][col] = 1 # c
        U[3][col] = 1 # d
        V[0*p + offset_k][col] = 1 # x
        W[offset_k*n + 1][col] = 1 # C21 (k=0, i=1)
        col += 1
        
        # m3 = a(y-w)
        U[0][col] = 1 # a
        V[0*p + offset_k+1][col] = 1 # y
        V[1*p + offset_k+1][col] = -1 # w
        W[offset_k*n + 0][col] = 1 # C11
        W[(offset_k+1)*n + 0][col] = 1 # C12 (k=1, i=0)
        col += 1
        
        # m4 = d(z-x)
        U[3][col] = 1 # d
        V[1*p + offset_k][col] = 1 # z
        V[0*p + offset_k][col] = -1 # x
        W[offset_k*n + 0][col] = 1 # C11
        W[offset_k*n + 1][col] = 1 # C21
        col += 1
        
        # m5 = (a+b)w
        U[0][col] = 1 # a
        U[1][col] = 1 # b (i=0,j=1) -> index 1
        V[1*p + offset_k+1][col] = 1 # w
        W[offset_k*n + 0][col] = -1 # C11
        W[(offset_k+1)*n + 0][col] = 1 # C12
        col += 1
        
        # m6 = (c-a)(x+z)
        U[2][col] = 1 # c
        U[0][col] = -1 # a
        V[0*p + offset_k][col] = 1 # x
        V[1*p + offset_k][col] = 1 # z
        W[(offset_k+1)*n + 1][col] = 1 # C22
        col += 1
        
        # m7 = (b-d)(y+w)
        U[1][col] = 1 # b
        U[3][col] = -1 # d
        V[0*p + offset_k+1][col] = 1 # y
        V[1*p + offset_k+1][col] = 1 # w
        W[offset_k*n + 0][col] = 1 # C11
        W[(offset_k+1)*n + 1][col] = -1 # C22
        col += 1

    # Block 1: k=0,1
    add_strassen_block(0)
    # Block 2: k=2,3
    add_strassen_block(2)
    
    # Block 3: k=4 (2x2x1)
    # C10 = a*x + b*y
    # C20 = c*x + d*y
    # Rank 4.
    # a*x
    U[0][col] = 1
    V[0*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    # b*y
    U[1][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    # c*x
    U[2][col] = 1
    V[0*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    # d*y
    U[3][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
