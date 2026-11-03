"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 103 decomposition using 2 Strassen blocks on disjoint J pairs
    # Block 1: I={0,1}, J={0,1}, K={0,1}
    # Block 2: I={0,1}, J={2,3}, K={2,3}
    # Rest: Schoolbook
    
    # We will build U, V, W matrices
    # Total rank = 7 + 7 + (3*5*7 - 2*8) = 14 + 89 = 103
    
    rank = 103
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i1, i2, j1, j2, k1, k2, start_col):
        # A submatrix: rows i1, i2; cols j1, j2
        # B submatrix: rows j1, j2; cols k1, k2
        # Strassen products:
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        # M6 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        # M7 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        
        # U entries (A side)
        # M1: A[i1,j1]=1, A[i2,j2]=1
        U[i1*m+j1][start_col] += 1
        U[i2*m+j2][start_col] += 1
        # M2: A[i2,j1]=1, A[i2,j2]=1
        U[i2*m+j1][start_col+1] += 1
        U[i2*m+j2][start_col+1] += 1
        # M3: A[i1,j1]=1
        U[i1*m+j1][start_col+2] += 1
        # M4: A[i2,j2]=1
        U[i2*m+j2][start_col+3] += 1
        # M5: A[i1,j1]=1, A[i1,j2]=1
        U[i1*m+j1][start_col+4] += 1
        U[i1*m+j2][start_col+4] += 1
        # M6: A[i1,j2]=1, A[i2,j2]=-1
        U[i1*m+j2][start_col+5] += 1
        U[i2*m+j2][start_col+5] -= 1
        # M7: A[i2,j1]=1, A[i1,j1]=-1
        U[i2*m+j1][start_col+6] += 1
        U[i1*m+j1][start_col+6] -= 1
        
        # V entries (B side)
        # M1: B[j1,k1]=1, B[j2,k2]=1
        V[j1*p+k1][start_col] += 1
        V[j2*p+k2][start_col] += 1
        # M2: B[j1,k1]=1
        V[j1*p+k1][start_col+1] += 1
        # M3: B[j1,k2]=1, B[j2,k2]=-1
        V[j1*p+k2][start_col+2] += 1
        V[j2*p+k2][start_col+2] -= 1
        # M4: B[j2,k1]=1, B[j1,k1]=-1
        V[j2*p+k1][start_col+3] += 1
        V[j1*p+k1][start_col+3] -= 1
        # M5: B[j2,k2]=1
        V[j2*p+k2][start_col+4] += 1
        # M6: B[j2,k1]=1, B[j2,k2]=1
        V[j2*p+k1][start_col+5] += 1
        V[j2*p+k2][start_col+5] += 1
        # M7: B[j1,k1]=1, B[j1,k2]=1
        V[j1*p+k1][start_col+6] += 1
        V[j1*p+k2][start_col+6] += 1
        
        # W entries (C side)
        # C[i1,k1] += M1 - M2 - M4 + M7
        W[k1*n+i1][start_col] += 1
        W[k1*n+i1][start_col+1] -= 1
        W[k1*n+i1][start_col+3] -= 1
        W[k1*n+i1][start_col+6] += 1
        
        # C[i1,k2] += M3 + M5
        W[k2*n+i1][start_col+2] += 1
        W[k2*n+i1][start_col+4] += 1
        
        # C[i2,k1] += M2 + M4 + M6
        W[k1*n+i2][start_col+1] += 1
        W[k1*n+i2][start_col+3] += 1
        W[k1*n+i2][start_col+5] += 1
        
        # C[i2,k2] += M1 - M3 - M5 + M6
        W[k2*n+i2][start_col] += 1
        W[k2*n+i2][start_col+2] -= 1
        W[k2*n+i2][start_col+4] -= 1
        W[k2*n+i2][start_col+5] += 1
        
        return start_col + 7

    # Add Block 1
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    
    # Add Block 2
    col = add_strassen(0, 1, 2, 3, 2, 3, col)
    
    # Add Schoolbook for the rest
    # We need to add 1 for each (i,j,k) NOT covered by Strassen blocks
    # Covered:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is part of a Strassen block
                covered = False
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    covered = True
                if i in [0,1] and j in [2,3] and k in [2,3]:
                    covered = True
                    
                if not covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
