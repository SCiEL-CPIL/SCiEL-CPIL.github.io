"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # This is provably optimal (lower bound 7, upper bound 7).
    #
    # Convention:
    #   U has shape (n*m, rank) = (4, 7)
    #   V has shape (m*p, rank) = (4, 7)
    #   W has shape (p*n, rank) = (4, 7)
    #
    # The tensor T[i, j, k, l] for multiplying A (i x j) by B (j x k) to get C (i x k)
    # is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, let me re-read the convention carefully.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    #
    # Standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # With U indexed by (i,j), V indexed by (j,k), W indexed by (k,i):
    # C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  ... no, that's not right either.
    #
    # Let me think about this more carefully. The matrix multiplication tensor T_{ijk} = delta_{jk} ... no.
    #
    # The standard formulation: we want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) ... no.
    #
    # Actually, the standard bilinear decomposition is:
    # C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    #
    # Hmm, that's not quite right either. Let me use the standard formulation.
    #
    # The matrix multiplication tensor is a 3-tensor T of shape (n*p, n*m, m*p) or similar.
    # The rank decomposition: T[a,b,c] = sum_r U[b,r] * V[c,r] * W[a,r]
    # where a indexes C (n*p entries), b indexes A (n*m entries), c indexes B (m*p entries).
    #
    # So: C_flat[a] = sum_r U[b, r] * V[c, r] * W[a, r] ... no.
    #
    # Let me be very precise. The bilinear form for matrix multiplication:
    # For each output entry C[i,k] (where i in [0,n), k in [0,p)):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # A rank-r decomposition writes this as:
    # C[i,k] = sum_{r=0}^{r-1} W[(k*n+i), r] * P_r(A) * Q_r(B)
    # where P_r(A) = sum_{i,j} U[(i*m+j), r] * A[i,j]
    # and Q_r(B) = sum_{j,k} V[(j*p+k), r] * B[j,k]
    #
    # So the full expression is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''])
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need the decomposition to be exact.
    #
    # Strassen's algorithm for 2x2 matrices uses 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now let me map this to the U, V, W format.
    #
    # A is 2x2: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # B is 2x2: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    # C is 2x2: C[0,0]=C00, C[0,1]=C01, C[1,0]=C10, C[1,1]=C11
    #
    # U has shape (4, 7): U[i*m+j, r] where i,j in {0,1}
    #   U[0, r] = coefficient of A00 in P_r
    #   U[1, r] = coefficient of A01 in P_r
    #   U[2, r] = coefficient of A10 in P_r
    #   U[3, r] = coefficient of A11 in P_r
    #
    # V has shape (4, 7): V[j*p+k, r] where j,k in {0,1}
    #   V[0, r] = coefficient of B00 in Q_r
    #   V[1, r] = coefficient of B01 in Q_r
    #   V[2, r] = coefficient of B10 in Q_r
    #   V[3, r] = coefficient of B11 in Q_r
    #
    # W has shape (4, 7): W[k*n+i, r] where k,i in {0,1}
    #   W[0, r] = coefficient for C00 (k=0, i=0)
    #   W[1, r] = coefficient for C01 (k=1, i=0)
    #   W[2, r] = coefficient for C10 (k=0, i=1)
    #   W[3, r] = coefficient for C11 (k=1, i=1)
    #
    # Strassen's 7 factors:
    # r=0: M1 = (A00+A11)*(B00+B11)
    #   U: A00=1, A01=0, A10=0, A11=1 -> U[:,0] = [1,0,0,1]
    #   V: B00=1, B01=0, B10=0, B11=1 -> V[:,0] = [1,0,0,1]
    #   W: C00 += M1, C01 += 0, C10 += 0, C11 += M1 -> W[:,0] = [1,0,0,1]
    #
    # r=1: M2 = (A10+A11)*B00
    #   U: A00=0, A01=0, A10=1, A11=1 -> U[:,1] = [0,0,1,1]
    #   V: B00=1, B01=0, B10=0, B11=0 -> V[:,1] = [1,0,0,0]
    #   W: C00 += 0, C01 += 0, C10 += M2, C11 -= M2 -> W[:,1] = [0,0,1,-1]
    #
    # r=2: M3 = A00*(B01-B11)
    #   U: A00=1, A01=0, A10=0, A11=0 -> U[:,2] = [1,0,0,0]
    #   V: B00=0, B01=1, B10=0, B11=-1 -> V[:,2] = [0,1,0,-1]
    #   W: C00 += 0, C01 += M3, C10 += 0, C11 += M3 -> W[:,2] = [0,1,0,1]
    #
    # r=3: M4 = A11*(B10-B00)
    #   U: A00=0, A01=0, A10=0, A11=1 -> U[:,3] = [0,0,0,1]
    #   V: B00=-1, B01=0, B10=1, B11=0 -> V[:,3] = [-1,0,1,0]
    #   W: C00 += M4, C01 += 0, C10 += M4, C11 += 0 -> W[:,3] = [1,0,1,0]
    #
    # r=4: M5 = (A00+A01)*B11
    #   U: A00=1, A01=1, A10=0, A11=0 -> U[:,4] = [1,1,0,0]
    #   V: B00=0, B01=0, B10=0, B11=1 -> V[:,4] = [0,0,0,1]
    #   W: C00 -= M5, C01 += M5, C10 += 0, C11 += 0 -> W[:,4] = [-1,1,0,0]
    #
    # r=5: M6 = (A10-A00)*(B00+B01)
    #   U: A00=-1, A01=0, A10=1, A11=0 -> U[:,5] = [-1,0,1,0]
    #   V: B00=1, B01=1, B10=0, B11=0 -> V[:,5] = [1,1,0,0]
    #   W: C00 += 0, C01 += 0, C10 += 0, C11 += M6 -> W[:,5] = [0,0,0,1]
    #
    # r=6: M7 = (A01-A11)*(B10+B11)
    #   U: A00=0, A01=1, A10=0, A11=-1 -> U[:,6] = [0,1,0,-1]
    #   V: B00=0, B01=0, B10=1, B11=1 -> V[:,6] = [0,0,1,1]
    #   W: C00 += M7, C01 += 0, C10 += 0, C11 += 0 -> W[:,6] = [1,0,0,0]
    #
    # Let me verify C00 = M1 + M4 - M5 + M7:
    #   W[0,:] = [1, 0, 0, 1, -1, 0, 1]
    #   Sum: M1 + 0 + 0 + M4 - M5 + 0 + M7 = M1 + M4 - M5 + M7 ✓
    #
    # C01 = M3 + M5:
    #   W[1,:] = [0, 0, 1, 0, 1, 0, 0]
    #   Sum: 0 + 0 + M3 + 0 + M5 + 0 + 0 = M3 + M5 ✓
    #
    # C10 = M2 + M4:
    #   W[2,:] = [0, 1, 0, 1, 0, 0, 0]
    #   Sum: 0 + M2 + 0 + M4 + 0 + 0 + 0 = M2 + M4 ✓
    #
    # C11 = M1 - M2 + M3 + M6:
    #   W[3,:] = [1, -1, 1, 0, 0, 1, 0]
    #   Sum: M1 - M2 + M3 + 0 + 0 + M6 + 0 = M1 - M2 + M3 + M6 ✓
    
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U columns (rows are A[0,0], A[0,1], A[1,0], A[1,1])
    U[0][0] = 1; U[3][0] = 1  # M1: A00+A11
    U[2][1] = 1; U[3][1] = 1  # M2: A10+A11
    U[0][2] = 1               # M3: A00
    U[3][3] = 1               # M4: A11
    U[0][4] = 1; U[1][4] = 1  # M5: A00+A01
    U[0][5] = -1; U[2][5] = 1 # M6: A10-A00
    U[1][6] = 1; U[3][6] = -1 # M7: A01-A11
    
    # V columns (rows are B[0,0], B[0,1], B[1,0], B[1,1])
    V[0][0] = 1; V[3][0] = 1  # M1: B00+B11
    V[0][1] = 1               # M2: B00
    V[1][2] = 1; V[3][2] = -1 # M3: B01-B11
    V[0][3] = -1; V[2][3] = 1 # M4: B10-B00
    V[3][4] = 1               # M5: B11
    V[0][5] = 1; V[1][5] = 1  # M6: B00+B01
    V[2][6] = 1; V[3][6] = 1  # M7: B10+B11
    
    # W columns (rows are C[0,0], C[0,1], C[1,0], C[1,1]) i.e. W[k*n+i]
    W[0][0] = 1; W[3][0] = 1  # M1: C00+=M1, C11+=M1
    W[2][1] = 1; W[3][1] = -1 # M2: C10+=M2, C11-=M2
    W[1][2] = 1; W[3][2] = 1  # M3: C01+=M3, C11+=M3
    W[0][3] = 1; W[2][3] = 1  # M4: C00+=M4, C10+=M4
    W[0][4] = -1; W[1][4] = 1 # M5: C00-=M5, C01+=M5
    W[3][5] = 1               # M6: C11+=M6
    W[0][6] = 1               # M7: C00+=M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
