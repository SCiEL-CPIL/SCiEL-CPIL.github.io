import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [2, 1]
    m_parts = [3, 2]
    p_parts = [3, 3]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                U_b, V_b, W_b = _construct_block(ni, mi, pi)
                U_blocks.append(U_b)
                V_blocks.append(V_b)
                W_blocks.append(W_b)
    
    U = np.hstack(U_blocks) if U_blocks else np.zeros((n*m, 0))
    V = np.hstack(V_blocks) if V_blocks else np.zeros((m*p, 0))
    W = np.hstack(W_blocks) if W_blocks else np.zeros((p*n, 0))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_block(n, m, p):
    """Construct factor matrices for n×m×p multiplication."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    else:
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen's algorithm for 2×2×2, rank 7."""
    # Strassen's 7 multiplications
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # The tensor decomposition has rank 7.
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)  
    # W has shape (2*2, 7) = (4, 7)
    
    # Indices: U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
    # n=m=p=2, so indices are 0..3 for each
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h): U rows for a,d; V rows for e,h; W rows for C11,C22
    # a is at index 0 (i=0,j=0), d is at index 3 (i=1,j=1)
    # e is at index 0 (j=0,k=0), h is at index 3 (j=1,k=1)
    # C11 is at W index k*n+i = 0*2+0 = 0, C22 is at W index 1*2+1 = 3
    
    # m1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1  # a + d
    V[0, 0] = 1; V[3, 0] = 1  # e + h
    W[0, 0] = 1; W[3, 0] = 1  # contributes to C11 and C22
    
    # m2 = (c+d)e: c at index 2 (i=1,j=0), d at index 3, e at index 0
    # Contributes to C21 (W index 1*2+0=2) and C22 (W index 3)
    U[2, 1] = 1; U[3, 1] = 1  # c + d
    V[0, 1] = 1  # e
    W[2, 1] = 1; W[3, 1] = -1  # C21 += m2, C22 -= m2
    
    # m3 = a(f-h): a at index 0, f at index 1 (j=0,k=1), h at index 3
    # Contributes to C12 (W index 0*2+1=1) and C22 (W index 3)
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f - h
    W[1, 2] = 1; W[3, 2] = 1  # C12 += m3, C22 += m3
    
    # m4 = d(g-e): d at index 3, g at index 2 (j=1,k=0), e at index 0
    # Contributes to C11 (W index 0) and C21 (W index 2)
    U[3, 3] = 1  # d
    V[2, 3] = 1; V[0, 3] = -1  # g - e
    W[0, 3] = 1; W[2, 3] = 1  # C11 += m4, C21 += m4
    
    # m5 = (a+b)h: a at index 0, b at index 1 (i=0,j=1), h at index 3
    # Contributes to C12 (W index 1) and C11 (W index 0) with -1
    U[0, 4] = 1; U[1, 4] = 1  # a + b
    V[3, 4] = 1  # h
    W[1, 4] = 1; W[0, 4] = -1  # C12 += m5, C11 -= m5
    
    # m6 = (c-a)(e+f): c at index 2, a at index 0, e at index 0, f at index 1
    # Contributes to C22 (W index 3)
    U[2, 5] = 1; U[0, 5] = -1  # c - a
    V[0, 5] = 1; V[1, 5] = 1  # e + f
    W[3, 5] = 1  # C22 += m6
    
    # m7 = (b-d)(g+h): b at index 1, d at index 3, g at index 2, h at index 3
    # Contributes to C11 (W index 0)
    U[1, 6] = 1; U[3, 6] = -1  # b - d
    V[2, 6] = 1; V[3, 6] = 1  # g + h
    W[0, 6] = 1  # C11 += m7
    
    return U, V, W
# EVOLVE-BLOCK-END
