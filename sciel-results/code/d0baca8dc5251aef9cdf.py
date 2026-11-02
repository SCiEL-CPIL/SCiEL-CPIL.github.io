"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # Strassen's 7 products
    # P1 = (a+d)(e+h)
    U[0][0] = 1  # a (0,0)
    U[3][0] = 1  # d (1,1)
    V[0][0] = 1  # e (0,0)
    V[3][0] = 1  # h (1,1)
    
    # P2 = (c+d)e
    U[2][1] = 1  # c (1,0)
    U[3][1] = 1  # d (1,1)
    V[0][1] = 1  # e (0,0)
    
    # P3 = a(f-h)
    U[0][2] = 1  # a (0,0)
    V[1][2] = 1  # f (0,1)
    V[3][2] = -1 # h (1,1)
    
    # P4 = d(g-e)
    U[3][3] = 1  # d (1,1)
    V[2][3] = 1  # g (1,0)
    V[0][3] = -1 # e (0,0)
    
    # P5 = (a+b)h
    U[0][4] = 1  # a (0,0)
    U[1][4] = 1  # b (0,1)
    V[3][4] = 1  # h (1,1)
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1  # c (1,0)
    U[0][5] = -1 # a (0,0)
    V[0][5] = 1  # e (0,0)
    V[1][5] = 1  # f (0,1)
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1  # b (0,1)
    U[3][6] = -1 # d (1,1)
    V[2][6] = 1  # g (1,0)
    V[3][6] = 1  # h (1,1)
    
    # W reconstruction for C[k*n+i] which is M[i,k] transposed to M[k,i]?
    # M[i,k] = sum_j A[i,j] B[j,k]
    # We established W[k*n+i] corresponds to M[i,k]
    # So W[0] (k=0,i=0) is M[0,0]
    # W[1] (k=0,i=1) is M[1,0]
    # W[2] (k=1,i=0) is M[0,1]
    # W[3] (k=1,i=1) is M[1,1]
    
    # M[0,0] = P1 + P4 - P5 + P7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # M[1,0] = P2 + P4
    W[1][1] = 1
    W[1][3] = 1
    
    # M[0,1] = P3 + P5
    W[2][2] = 1
    W[2][4] = 1
    
    # M[1,1] = P1 - P2 + P3 + P6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
