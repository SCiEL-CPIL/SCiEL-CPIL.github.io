"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 35 (5 blocks of Strassen rank 7)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for each of the 7 products
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # For each block b (0..4), we have 7 terms
    for b in range(5):
        # Columns of B in this block are 2*b and 2*b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 1: (a+d)(e+h)
        r = b * 7 + 0
        U[0][r] += 1  # a
        U[3][r] += 1  # d
        V[10*0 + k0][r] += 1  # e
        V[10*1 + k1][r] += 1  # h
        W[2*k0 + 0][r] += 1  # C[0, k0]
        W[2*k1 + 1][r] += 1  # C[1, k1]
        
        # Term 2: (c+d)e
        r = b * 7 + 1
        U[2][r] += 1  # c
        U[3][r] += 1  # d
        V[10*0 + k0][r] += 1  # e
        W[2*k0 + 1][r] += 1  # C[1, k0]
        W[2*k1 + 1][r] += -1 # C[1, k1]
        
        # Term 3: a(f-h)
        r = b * 7 + 2
        U[0][r] += 1  # a
        V[10*0 + k1][r] += 1  # f
        V[10*1 + k1][r] += -1 # -h
        W[2*k1 + 0][r] += 1  # C[0, k1]
        W[2*k1 + 1][r] += 1  # C[1, k1]
        
        # Term 4: d(g-e)
        r = b * 7 + 3
        U[3][r] += 1  # d
        V[10*1 + k0][r] += 1  # g
        V[10*0 + k0][r] += -1 # -e
        W[2*k0 + 0][r] += 1  # C[0, k0]
        W[2*k0 + 1][r] += 1  # C[1, k0]
        
        # Term 5: (a+b)h
        r = b * 7 + 4
        U[0][r] += 1  # a
        U[1][r] += 1  # b
        V[10*1 + k1][r] += 1  # h
        W[2*k1 + 0][r] += 1  # C[0, k1]
        W[2*k0 + 0][r] += -1 # C[0, k0]
        
        # Term 6: (c-a)(e+f)
        r = b * 7 + 5
        U[2][r] += 1  # c
        U[0][r] += -1 # -a
        V[10*0 + k0][r] += 1  # e
        V[10*0 + k1][r] += 1  # f
        W[2*k1 + 1][r] += 1  # C[1, k1]
        
        # Term 7: (b-d)(g+h)
        r = b * 7 + 6
        U[1][r] += 1  # b
        U[3][r] += -1 # -d
        V[10*1 + k0][r] += 1  # g
        V[10*1 + k1][r] += 1  # h
        W[2*k0 + 0][r] += 1  # C[0, k0]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
