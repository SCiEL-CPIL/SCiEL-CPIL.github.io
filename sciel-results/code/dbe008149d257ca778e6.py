"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=4 into two channels of size 2
    # Sub-problem 1: (n=2, m1=2, p=8) -> uses rows 0,1 of intermediate
    # Sub-problem 2: (n=2, m2=2, p=8) -> uses rows 2,3 of intermediate
    # Each sub-problem: trivial rank = 2*2*8 = 32, total = 64
    # But we can do better with a structured decomposition.
    # Use Kronecker product approach: (2,2,2) x (1,2,4)
    # (2,2,2) via Strassen: rank 7
    # (1,2,4) trivial: rank 8
    # Total rank: 7 * 8 = 56
    
    # Let's build the (2,2,2) Strassen decomposition
    # Strassen for 2x2 x 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For (1,2,4): 1x2 x 2x4 -> 1x4
    # C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0..3
    # Trivial: 8 terms
    
    # Kronecker: total rank = 7 * 8 = 56
    rank_total = 56
    U = [[0] * rank_total for _ in range(n * m)]  # (2*4, 56) = (8, 56)
    V = [[0] * rank_total for _ in range(m * p)]  # (4*8, 56) = (32, 56)
    W = [[0] * rank_total for _ in range(p * n)]  # (8*2, 56) = (16, 56)
    
    # Strassen factors for 2x2 x 2x2
    # U_strassen: (4, 7) where index is (i*2+j)
    # V_strassen: (4, 7) where index is (j*2+k)
    # W_strassen: (4, 7) where index is (k*2+i)
    
    # M1: (A11+A22)*(B11+B22)
    # A11=idx0, A22=idx3, B11=idx0, B22=idx3
    # M2: (A21+A22)*B11
    # A21=idx1, A22=idx3, B11=idx0
    # M3: A11*(B12-B22)
    # A11=idx0, B12=idx1, B22=idx3
    # M4: A22*(B21-B11)
    # A22=idx3, B21=idx2, B11=idx0
    # M5: (A11+A12)*B22
    # A11=idx0, A12=idx2, B22=idx3
    # M6: (A21-A11)*(B11+B12)
    # A21=idx1, A11=idx0, B11=idx0, B12=idx1
    # M7: (A12-A22)*(B21+B22)
    # A12=idx2, A22=idx3, B21=idx2, B22=idx3
    
    # Strassen U (for A matrix, 2x2, index i*2+j):
    # M1: A11+A22 -> [1,0,0,1]
    # M2: A21+A22 -> [0,1,0,1]
    # M3: A11 -> [1,0,0,0]
    # M4: A22 -> [0,0,0,1]
    # M5: A11+A12 -> [1,0,1,0]
    # M6: A21-A11 -> [-1,1,0,0]
    # M7: A12-A22 -> [0,0,1,-1]
    
    # Strassen V (for B matrix, 2x2, index j*2+k):
    # M1: B11+B22 -> [1,0,0,1]
    # M2: B11 -> [1,0,0,0]
    # M3: B12-B22 -> [0,1,0,-1]
    # M4: B21-B11 -> [0,0,1,-1]  # wait, B21 is idx 2, B11 is idx 0
    # Let me re-index: B[j*2+k], so B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # M1: B11+B22 = B[0,0]+B[1,1] -> [1,0,0,1]
    # M2: B11 = B[0,0] -> [1,0,0,0]
    # M3: B12-B22 = B[0,1]-B[1,1] -> [0,1,0,-1]
    # M4: B21-B11 = B[1,0]-B[0,0] -> [-1,0,1,0]
    # M5: B22 = B[1,1] -> [0,0,0,1]
    # M6: B11+B12 = B[0,0]+B[0,1] -> [1,1,0,0]
    # M7: B21+B22 = B[1,0]+B[1,1] -> [0,0,1,1]
    
    # Strassen W (for C matrix, 2x2, index k*2+i):
    # C[k*2+i], so C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # C11 = C[0,0] = M1+M4-M5+M7
    # C12 = C[0,1] = M3+M5
    # C21 = C[1,0] = M2+M4
    # C22 = C[1,1] = M1-M2+M3+M6
    
    # W_strassen[col][k*2+i]:
    # M1 (col 0): C[0,0]+=1, C[1,1]+=1 -> W[0][0]=1, W[3][0]=1
    # M2 (col 1): C[1,0]+=1, C[1,1]-=1 -> W[2][1]=1, W[3][1]=-1
    # M3 (col 2): C[0,1]+=1, C[1,1]+=1 -> W[1][2]=1, W[3][2]=1
    # M4 (col 3): C[0,0]+=1, C[1,0]+=1 -> W[0][3]=1, W[2][3]=1
    # M5 (col 4): C[0,0]-=1, C[0,1]+=1 -> W[0][4]=-1, W[1][4]=1
    # M6 (col 5): C[1,1]+=1 -> W[3][5]=1
    # M7 (col 6): C[0,0]+=1 -> W[0][6]=1
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [1, 1, 0, 1, 0, 0, -1],   # A22
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1],   # B22
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # C22 (k=1,i=1)
    ]
    
    # For (1,2,4): 1x2 x 2x4 -> 1x4
    # A has 1 element (index 0), B has 2x4=8 elements, C has 1x4=4 elements
    # C[k] = A[0] * (B[0,k] + B[1,k])? No, C[k] = A[0]*B[0,k] + A[0]*B[1,k]? 
    # Wait, 1x2 x 2x4: C[i,k] = sum_j A[i,j]*B[j,k]. Here i is always 0.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # Trivial: 8 terms (2*4)
    # U_t: (1, 8): A[0,0] for k=0..3 with B[0,k], A[0,1] for k=0..3 with B[1,k]
    # Actually: for each (j,k), term: U[0][j*4+k] = A[0,j], V[j*4+k][j*4+k] = B[j,k], W[k][j*4+k] = 1
    
    # U_t: shape (1, 8)
    U_t = [[1, 0, 1, 0, 1, 0, 1, 0]]  # A[0,0] for even k, A[0,1] for odd k? No...
    # Let me index: j=0,1; k=0,1,2,3. Index = j*4+k
    # For j=0: A[0,0] multiplies B[0,k] for all k
    # For j=1: A[0,1] multiplies B[1,k] for all k
    U_t = [[1, 1, 1, 1, 0, 0, 0, 0]]  # j=0: A[0,0]=1 for k=0..3, j=1: A[0,1]=0? No...
    # U_t[row][col] where row is A index (only 0), col is (j*4+k)
    # U_t[0][j*4+k] = 1 if we're using A[0,j]
    # For j=0: U_t[0][0..3] = 1 (A[0,0])
    # For j=1: U_t[0][4..7] = 1 (A[0,1])
    U_t = [[1, 1, 1, 1, 1, 1, 1, 1]]  # A[0,j] is just A[0] for j=0,1? No, A is 1x2, so A[0,0] and A[0,1]
    
    # Hmm, I think I'm confusing myself. Let me restart with a simpler approach.
    
    # Just use the trivial decomposition but structured as two halves.
    # This gives rank 64, same as before, but let's make sure it's correct.
    # Actually, let me just use the original trivial decomposition which is known correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
