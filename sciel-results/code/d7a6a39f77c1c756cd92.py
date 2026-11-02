"""Matrix multiplication 2,2,14 with rank-minimal decomposition using Strassen blocks."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # For each pair of columns in B, we can share some computations
    # But actually, for 2x2 by 2x14, we can process column pairs
    
    # The key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    # For 2x2 A and 2x14 B, we have 4*14 = 56 naive multiplications
    # Using Strassen per column pair: 7*7 = 49 operations
    
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For B with 14 columns, we group into 7 pairs: (0,1), (2,3), ..., (12,13)
    # For each pair (2k, 2k+1), we apply Strassen
    
    rank = 7 * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[j,k] = j*14 + k
    # C indices: C[k,i] = k*2 + i (trace convention W[k*n+i,r])
    
    for block in range(7):
        k0 = 2 * block      # first column in pair
        k1 = 2 * block + 1  # second column in pair
        
        # For this block, we have 7 Strassen terms
        # B00 = B[0, k0], B01 = B[0, k1]
        # B10 = B[1, k0], B11 = B[1, k1]
        
        # Term 1: M1 = (A00+A11)*(B00+B11)
        col = block * 7 + 0
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[0 * p + k0][col] = 1   # B[0,k0]
        V[1 * p + k1][col] = 1   # B[1,k1]
        # C00 += M1, C11 += M1
        W[k0 * n + 0][col] = 1   # C[k0, 0]
        W[k1 * n + 1][col] = 1   # C[k1, 1]
        
        # Term 2: M2 = (A10+A11)*B00
        col = block * 7 + 1
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[0 * p + k0][col] = 1   # B[0,k0]
        # C10 += M2
        W[k0 * n + 1][col] = 1   # C[k0, 1]
        
        # Term 3: M3 = A00*(B01-B11)
        col = block * 7 + 2
        U[0][col] = 1   # A00
        V[0 * p + k1][col] = 1   # B[0,k1]
        V[1 * p + k1][col] = -1  # -B[1,k1]
        # C01 += M3, C11 += M3
        W[k1 * n + 0][col] = 1   # C[k1, 0]
        W[k1 * n + 1][col] = 1   # C[k1, 1]
        
        # Term 4: M4 = A11*(B10-B00)
        col = block * 7 + 3
        U[3][col] = 1   # A11
        V[1 * p + k0][col] = 1   # B[1,k0]
        V[0 * p + k0][col] = -1  # -B[0,k0]
        # C00 += M4, C10 += M4
        W[k0 * n + 0][col] = 1   # C[k0, 0]
        W[k0 * n + 1][col] = 1   # C[k0, 1]
        
        # Term 5: M5 = (A00+A01)*B11
        col = block * 7 + 4
        U[0][col] = 1   # A00
        U[1][col] = 1   # A01
        V[1 * p + k1][col] = 1   # B[1,k1]
        # C00 -= M5, C01 += M5
        W[k0 * n + 0][col] = -1  # -C[k0, 0]
        W[k1 * n + 0][col] = 1   # C[k1, 0]
        
        # Term 6: M6 = (A10-A00)*(B00+B01)
        col = block * 7 + 5
        U[2][col] = 1   # A10
        U[0][col] = -1  # -A00
        V[0 * p + k0][col] = 1   # B[0,k0]
        V[0 * p + k1][col] = 1   # B[0,k1]
        # C11 += M6
        W[k1 * n + 1][col] = 1   # C[k1, 1]
        
        # Term 7: M7 = (A01-A11)*(B10+B11)
        col = block * 7 + 6
        U[1][col] = 1   # A01
        U[3][col] = -1  # -A11
        V[1 * p + k0][col] = 1   # B[1,k0]
        V[1 * p + k1][col] = 1   # B[1,k1]
        # C00 += M7, C11 -= M7
        W[k0 * n + 0][col] = 1   # C[k0, 0]
        W[k1 * n + 1][col] = -1  # -C[k1, 1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
