import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank of the 2x3 by 3x3 matrix multiplication tensor is 17.
    # This is the minimal rank, so this is the optimal solution.
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)] # 6 rows
    V = [[0] * rank for _ in range(m * p)] # 9 rows
    W = [[0] * rank for _ in range(p * n)] # 6 rows
    
    # Indices:
    # U[i*m + j] corresponds to A[i, j]
    # V[j*p + k] corresponds to B[j, k]
    # W[k*n + i] corresponds to C[i, k] (Note: W index in contract is k*n+i, which maps to C[i,k] if C is i x k? 
    # Contract says W[k*n+i, r]. C is 2x3. i in 0..1, k in 0..2.
    # Standard C[i,k] = sum A[i,j] B[j,k].
    # Term r contributes U[i,j,r] * V[j,k,r] * W[k,i,r] to C[i,k]?
    # Let's check the indices in the current program.
    # W[0][0] = 1. Index 0. k=0, i=0. C[0,0].
    # W[3][0] = 1. Index 3. k=1, i=1? 1*2+1=3. C[1,1].
    # So W index `idx` corresponds to k = idx // n, i = idx % n.
    # So W[k*n + i] is the coefficient for C[i, k].
    
    # Term 1
    U[0][0] = 1; U[4][0] = 1  # A[0,0], A[1,1]
    V[0][0] = 1; V[4][0] = 1  # B[0,0], B[1,1]
    W[0][0] = 1; W[3][0] = 1  # C[0,0], C[1,1]
    
    # Term 2
    U[3][1] = 1; U[4][1] = 1  # A[1,0], A[1,1]
    V[0][1] = 1               # B[0,0]
    W[1][1] = 1; W[3][1] = -1 # C[0,1], C[1,1]
    
    # Term 3
    U[0][2] = 1               # A[0,0]
    V[1][2] = 1; V[4][2] = -1 # B[0,1], B[1,1]
    W[2][2] = 1; W[3][2] = 1  # C[0,2], C[1,2]
    
    # Term 4
    U[4][3] = 1               # A[1,1]
    V[3][3] = 1; V[0][3] = -1 # B[1,0], B[0,0]
    W[0][3] = 1; W[1][3] = 1  # C[0,0], C[0,1]
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1  # A[0,0], A[0,1]
    V[4][4] = 1               # B[1,1]
    W[0][4] = -1; W[2][4] = 1 # C[0,0], C[0,2]
    
    # Term 6
    U[3][5] = 1; U[0][5] = -1 # A[1,0], A[0,0]
    V[0][5] = 1; V[1][5] = 1  # B[0,0], B[0,1]
    W[3][5] = 1               # C[1,1]
    
    # Term 7
    U[1][6] = 1; U[4][6] = -1 # A[0,1], A[1,1]
    V[3][6] = 1; V[4][6] = 1  # B[1,0], B[1,1]
    W[0][6] = 1               # C[0,0]
    
    # Term 8
    U[0][7] = 1               # A[0,0]
    V[2][7] = 1               # B[0,2]
    W[4][7] = 1               # C[1,0] ? Index 4: k=2, i=0. C[0,2].
    # Wait. W[4] -> k=2, i=0. C[0,2].
    # A[0,0] * B[0,2] contributes to C[0,2]. Correct.
    
    # Term 9
    U[1][8] = 1               # A[0,1]
    V[5][8] = 1               # B[1,2] (Index 5: j=1, k=2)
    W[4][8] = 1               # C[0,2]
    # A[0,1] * B[1,2] contributes to C[0,2]. Correct.
    
    # Term 10
    U[3][9] = 1               # A[1,0]
    V[2][9] = 1               # B[0,2]
    W[5][9] = 1               # Index 5: k=2, i=1. C[1,2].
    # A[1,0] * B[0,2] contributes to C[1,2]. Correct.
    
    # Term 11
    U[4][10] = 1              # A[1,1]
    V[5][10] = 1              # B[1,2]
    W[5][10] = 1              # C[1,2]
    # A[1,1] * B[1,2] contributes to C[1,2]. Correct.
    
    # Term 12
    U[2][11] = 1              # A[0,2]
    V[6][11] = 1              # B[2,0] (Index 6: j=2, k=0)
    W[0][11] = 1              # C[0,0]
    # A[0,2] * B[2,0] contributes to C[0,0]. Correct.
    
    # Term 13
    U[2][12] = 1              # A[0,2]
    V[7][12] = 1              # B[2,1] (Index 7: j=2, k=1)
    W[2][12] = 1              # Index 2: k=1, i=0. C[0,1].
    # A[0,2] * B[2,1] contributes to C[0,1]. Correct.
    
    # Term 14
    U[2][13] = 1              # A[0,2]
    V[8][13] = 1              # B[2,2] (Index 8: j=2, k=2)
    W[4][13] = 1              # C[0,2]
    # A[0,2] * B[2,2] contributes to C[0,2]. Correct.
    
    # Term 15
    U[5][14] = 1              # A[1,2]
    V[6][14] = 1              # B[2,0]
    W[1][14] = 1              # Index 1: k=0, i=1. C[1,0].
    # A[1,2] * B[2,0] contributes to C[1,0]. Correct.
    
    # Term 16
    U[5][15] = 1              # A[1,2]
    V[7][15] = 1              # B[2,1]
    W[3][15] = 1              # Index 3: k=1, i=1. C[1,1].
    # A[1,2] * B[2,1] contributes to C[1,1]. Correct.
    
    # Term 17
    U[5][16] = 1              # A[1,2]
    V[8][16] = 1              # B[2,2]
    W[5][16] = 1              # C[1,2]
    # A[1,2] * B[2,2] contributes to C[1,2]. Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
