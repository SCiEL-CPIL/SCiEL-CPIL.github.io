"""Strassen rank-7 decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # m1 = (a-d) * (e+f)
    # m2 = (c+d) * e
    # m3 = a * (f-e)
    # m4 = d * (h-g)
    # m5 = (a+b) * h
    # m6 = (c-a) * (e+f)
    # m7 = (b-d) * (g+h)
    #
    # C[0,0] = m1 + m4 + m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # Tensor decomposition: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Here n=m=p=2, so:
    # U has shape (4, 7), rows indexed by (i,j) where i=row of A, j=col of A
    #   U[i*2+j, r] = coefficient of A[i,j] in product r
    # V has shape (4, 7), rows indexed by (j,k) where j=row of B, k=col of B  
    #   Wait, let me re-read: V[j*p+k, r] where j is... 
    #   Actually the convention is:
    #   U[i*m+j, r] - i indexes rows of A (0..n-1), j indexes cols of A (0..m-1)
    #   V[j*p+k, r] - j indexes rows of B (0..m-1), k indexes cols of B (0..p-1)
    #   W[k*n+i, r] - k indexes cols of B / rows of C... 
    #   
    #   The matrix multiplication tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    #   So T[i,j,k] is the coefficient relating A[i,j], B[j,k] to C[i,k]
    #   
    #   Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    #   For 2x2x2:
    #   U has 4 rows (i,j) in {0,1}x{0,1}, V has 4 rows (j,k) in {0,1}x{0,1}, W has 4 rows (k,i) in {0,1}x{0,1}
    #
    #   Let's verify with Strassen:
    #   The tensor T[i,j,k] is 1 for all valid (i,j,k) since it's just the standard matmul.
    #   
    #   For Strassen, we express each product m_r as a linear combination of A entries times B entries:
    #   m1 = (a-d)*(e+f) = a*e + a*f - d*e - d*f
    #       A coefficients: a=1, b=0, c=0, d=-1
    #       B coefficients: e=1, f=1, g=0, h=0
    #   m2 = (c+d)*e
    #       A coefficients: a=0, b=0, c=1, d=1
    #       B coefficients: e=1, f=0, g=0, h=0
    #   m3 = a*(f-e)
    #       A coefficients: a=1, b=0, c=0, d=0
    #       B coefficients: e=-1, f=1, g=0, h=0
    #   m4 = d*(h-g)
    #       A coefficients: a=0, b=0, c=0, d=1
    #       B coefficients: e=0, f=0, g=-1, h=1
    #   m5 = (a+b)*h
    #       A coefficients: a=1, b=1, c=0, d=0
    #       B coefficients: e=0, f=0, g=0, h=1
    #   m6 = (c-a)*(e+f)
    #       A coefficients: a=-1, b=0, c=1, d=0
    #       B coefficients: e=1, f=1, g=0, h=0
    #   m7 = (b-d)*(g+h)
    #       A coefficients: a=0, b=1, c=0, d=-1
    #       B coefficients: e=0, f=0, g=1, h=1
    #
    #   U rows (A coefficients, row index = i*2+j):
    #   Row 0 (i=0,j=0, entry a): [1, 0, 1, 0, 1, -1, 0]
    #   Row 1 (i=0,j=1, entry b): [0, 0, 0, 0, 1, 0, 1]
    #   Row 2 (i=1,j=0, entry c): [0, 1, 0, 0, 0, 1, 0]
    #   Row 3 (i=1,j=1, entry d): [-1, 1, 0, 1, 0, 0, -1]
    #
    #   V rows (B coefficients, row index = j*2+k):
    #   Row 0 (j=0,k=0, entry e): [1, 1, -1, 0, 0, 1, 0]
    #   Row 1 (j=0,k=1, entry f): [1, 0, 1, 0, 0, 1, 0]
    #   Row 2 (j=1,k=0, entry g): [0, 0, 0, -1, 0, 0, 1]
    #   Row 3 (j=1,k=1, entry h): [0, 0, 0, 1, 1, 0, 1]
    #
    #   W rows (C coefficients, row index = k*2+i):
    #   C[0,0] = m1 + m4 + m5 + m7
    #   C[0,1] = m3 + m5
    #   C[1,0] = m2 + m4
    #   C[1,1] = m1 - m2 + m3 + m6
    #   
    #   W row index = k*n+i = k*2+i:
    #   Row 0 (k=0,i=0, C[0,0]): [1, 0, 0, 1, 1, 0, 1]
    #   Row 1 (k=0,i=1, C[1,0]): [0, 1, 0, 1, 0, 0, 0]
    #   Row 2 (k=1,i=0, C[0,1]): [0, 0, 1, 0, 1, 0, 0]
    #   Row 3 (k=1,i=1, C[1,1]): [1, -1, 1, 0, 0, 1, 0]
    
    U = [[1, 0, 1, 0, 1, -1, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, 1, 0],
         [-1, 1, 0, 1, 0, 0, -1]]
    
    V = [[1, 1, -1, 0, 0, 1, 0],
         [1, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, -1, 0, 0, 1],
         [0, 0, 0, 1, 1, 0, 1]]
    
    W = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
