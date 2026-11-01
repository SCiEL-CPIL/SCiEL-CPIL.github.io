"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7 for 2x2 matrix multiplication.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # n=m=p=2
    # W index: k*n+i
    # U index: i*m+j
    # V index: j*p+k
    
    # Strassen's 7 products for 2x2 matrices:
    # Let A = [[a11, a12], [a21, a22]] and B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    
    # Standard Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2*2 = 4 rows
    V = [[0] * rank for _ in range(4)]  # 2*2 = 4 rows
    W = [[0] * rank for _ in range(4)]  # 2*2 = 4 rows
    
    # U maps input A indices (i,j) to factor c
    # V maps input B indices (j,k) to factor c
    # W maps output C indices (k,i) to factor c
    
    # For each of the 7 Strassen products, we need linear combinations
    # of A and B entries, and then linear combinations of products to form C.
    
    # Product 1: (a11+a22)(b11+b22)
    # U contributions: a11 -> U[0,0]=1, a22 -> U[3,0]=1
    # V contributions: b11 -> V[0,0]=1, b22 -> V[3,0]=1
    # W contributions: c11 -> W[1,0]=1 (k=0,i=0), c12 -> W[2,0]=1 (k=0,i=1)? 
    # Wait, W[k*n+i, c]. For 2x2, n=2.
    # c11 is at i=0,k=0 -> W[0*2+0]=W[0]
    # c12 is at i=0,k=1 -> W[1*2+0]=W[2]
    # c21 is at i=1,k=0 -> W[0*2+1]=W[1]
    # c22 is at i=1,k=1 -> W[1*2+1]=W[3]
    
    # Let's re-index carefully:
    # A[i,j] where i,j in {0,1}
    # B[j,k] where j,k in {0,1}
    # C[i,k] = sum_j A[i,j]B[j,k]
    
    # U row index: i*m+j = i*2+j
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[2]
    # A[1,1] -> U[3]
    
    # V row index: j*p+k = j*2+k
    # B[0,0] -> V[0]
    # B[0,1] -> V[1]
    # B[1,0] -> V[2]
    # B[1,1] -> V[3]
    
    # W row index: k*n+i = k*2+i
    # C[0,0] (i=0,k=0) -> W[0*2+0]=W[0]
    # C[0,1] (i=0,k=1) -> W[1*2+0]=W[2]
    # C[1,0] (i=1,k=0) -> W[0*2+1]=W[1]
    # C[1,1] (i=1,k=1) -> W[1*2+1]=W[3]
    
    # Strassen products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Column 0: m1 = (a11+a22)(b11+b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # c11
    W[3][0] = 1  # c22
    
    # Column 1: m2 = (a21+a22)b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[1][1] = 1  # c21
    W[3][1] = -1 # c22
    
    # Column 2: m3 = a11(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1 # b22
    W[2][2] = 1  # c12
    W[3][2] = 1  # c22
    
    # Column 3: m4 = a22(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1 # b11
    W[0][3] = 1  # c11
    W[1][3] = 1  # c21
    
    # Column 4: m5 = (a11+a12)b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][4] = -1 # c11
    W[2][4] = 1  # c12
    
    # Column 5: m6 = (a21-a11)(b11+b12)
    U[2][5] = 1  # a21
    U[0][5] = -1 # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][5] = 1  # c22
    
    # Column 6: m7 = (a12-a22)(b21+b22)
    U[1][6] = 1  # a12
    U[3][6] = -1 # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][6] = 1  # c11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
