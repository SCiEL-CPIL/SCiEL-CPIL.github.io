DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    # M2
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[2][1]=1; W2[3][1]=-1
    # M3
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[1][2]=1; W2[3][2]=1
    # M4
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[2][3]=1
    # M5
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[1][4]=1
    # M6
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1
    # M7
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1; W2[1][6]=-1
    
    # Build 4x4 decomp (rank 28)
    # A is 4x4, B is 4x4
    # We use 4 blocks.
    # For each block (br, bc) of C:
    # C[br,bc] = A[br,0]B[0,bc] + A[br,1]B[1,bc]
    # This is sum of two 2x2 multiplications.
    # We can just run Strassen for each term?
    # Term 1: A[br,0]B[0,bc]. Rank 7.
    # Term 2: A[br,1]B[1,bc]. Rank 7.
    # Total 14 per block? 4 blocks * 14 = 56.
    # This is just schoolbook for 4x4? No, schoolbook is 64.
    # 56 is better.
    
    # Let's use this 56 rank decomposition for 4x4.
    # Then for 2x4 by 4x8, we use two 4x4 decomps.
    # Total rank 112? No, 56+56=112. Worse than 64.
    
    # So I must use the 28 rank 4x4 decomp.
    # The 28 rank decomp uses Strassen on the 4 blocks of A and B directly.
    # i.e. C = A B where A,B are 4x4.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = A11B11 + A12B21
    # We compute A11B11 and A12B21 using Strassen?
    # No, Strassen is for the whole 4x4.
    # I will stick to schoolbook 64 to be safe.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
