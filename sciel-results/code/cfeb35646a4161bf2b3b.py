"""Tensor decomposition for 2x2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-11 decomposition of the 2x2x3 matrix multiplication tensor.
    
    Uses Strassen's 7-term decomposition for the first two columns (2x2x2 part)
    and 4 additional terms for the third column (2x2x1 part).
    
    Index convention:
    - A is 2x2: A[i,j] for i in {0,1}, j in {0,1}
    - B is 2x3: B[j,k] for j in {0,1}, k in {0,1,2}
    - C is 2x3: C[i,k] for i in {0,1}, k in {0,1,2}
    
    U shape: (n*m, rank) = (4, 11)  indexed by (i,j)
    V shape: (m*p, rank) = (6, 11)  indexed by (j,k)
    W shape: (p*n, rank) = (6, 11)  indexed by (k,i) per W[k*n+i, r]
    
    For each rank-1 term r, C[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where the sum is over all j (the inner dimension of the multiplication).
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    U = [[0] * rank for _ in range(n * m)]  # (4, 11)
    V = [[0] * rank for _ in range(m * p)]  # (6, 11)
    W = [[0] * rank for _ in range(p * n)]  # (6, 11)
    
    # Strassen 7 terms for the 2x2x2 part (columns 0,1 of B)
    # Term 0: M1 = (a00 + a11)(b00 + b11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+0][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    W[0*2+1][0] = 1; W[1*2+0][0] = 1
    
    # Term 1: M2 = (a10 + a11)(b00)
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*3+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = 1
    
    # Term 2: M3 = a00(b01 - b11)
    U[0*2+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[0*2+0][2] = 1; W[1*2+0][2] = 1
    
    # Term 3: M4 = a11(b10 - b00)
    U[1*2+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[0*2+1][3] = 1; W[1*2+1][3] = 1
    
    # Term 4: M5 = (a00 + a01)(b11)
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*3+1][4] = 1
    W[1*2+0][4] = 1; W[1*2+1][4] = 1
    
    # Term 5: M6 = (a10 - a00)(b00 + b01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*3+0][5] = 1; V[0*3+1][5] = 1
    W[0*2+0][5] = 1; W[0*2+1][5] = 1
    
    # Term 6: M7 = (a01 - a11)(b10 + b11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*3+0][6] = 1; V[1*3+1][6] = 1
    W[0*2+1][6] = 1; W[1*2+0][6] = 1
    
    # 4 additional terms for the third column (column 2 of B)
    # Term 7: a00 * b02
    U[0*2+0][7] = 1
    V[0*3+2][7] = 1
    W[2*2+0][7] = 1  # k=2, i=0
    
    # Term 8: a01 * b12
    U[0*2+1][8] = 1
    V[1*3+2][8] = 1
    W[2*2+0][8] = 1  # k=2, i=0
    
    # Term 9: a10 * b02
    U[1*2+0][9] = 1
    V[0*3+2][9] = 1
    W[2*2+1][9] = 1  # k=2, i=1
    
    # Term 10: a11 * b12
    U[1*2+1][10] = 1
    V[1*3+2][10] = 1
    W[2*2+1][10] = 1  # k=2, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
