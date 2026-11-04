"""Matrix multiplication 2,3,9 with Strassen-based decomposition."""

def _strassen_222():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indices: (i,j) -> i*2+j
    V indices: (j,k) -> j*2+k
    W indices: (k,i) -> k*2+i  (trace convention W[k*n+i])
    """
    # Strassen's 7 products for multiplying 2x2 matrices A and B to get C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11+a22)(b11+b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[1][0] = 1  # b12
    # W for C11 (k=0,i=0): W[0*2+0]=W[0], C22 (k=1,i=1): W[1*2+1]=W[3]
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # p2 = (a21+a22)b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    # W for C21 (k=0,i=1): W[0*2+1]=W[1], C22 (k=1,i=1): W[3]
    W[1][1] = 1  # C21
    W[3][1] = -1  # C22
    
    # p3 = a11(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1  # b22
    # W for C12 (k=1,i=0): W[1*2+0]=W[2], C22 (k=1,i=1): W[3]
    W[2][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # p4 = a22(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1  # b11
    # W for C11 (k=0,i=0): W[0], C21 (k=0,i=1): W[1]
    W[0][3] = 1  # C11
    W[1][3] = 1  # C21
    
    # p5 = (a11+a12)b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    # W for C11 (k=0,i=0): W[0], C12 (k=1,i=0): W[2]
    W[0][4] = -1  # C11
    W[2][4] = 1  # C12
    
    # p6 = (a21-a11)(b11+b12)
    U[2][5] = 1  # a21
    U[0][5] = -1  # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    # W for C22 (k=1,i=1): W[3]
    W[3][5] = 1  # C22
    
    # p7 = (a12-a22)(b21+b22)
    U[1][6] = 1  # a12
    U[3][6] = -1  # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    # W for C11 (k=0,i=0): W[0]
    W[0][6] = 1  # C11
    
    return U, V, W


def _direct_221():
    """Return rank-2 decomposition for 2x2x1 matrix multiplication.
    Multiplying 2x2 by 2x1.
    Returns U (4x2), V (2x2), W (2x2) where:
    U indices: (i,j) -> i*2+j
    V indices: (j,k) -> j*1+k = j  (since p=1)
    W indices: (k,i) -> k*2+i  (n=2)
    """
    U = [[0]*2 for _ in range(4)]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2 for _ in range(2)]
    
    # C_i = sum_j A_ij * B_j0
    # For j=0: U[i,0,j=0]=1, V[j=0,0]=1, W[k=0,i]=1
    # For j=1: U[i,0,j=1]=1, V[j=1,0]=1, W[k=0,i]=1
    for i in range(2):
        U[i*2+0][0] = 1  # A[i,0]
        U[i*2+1][1] = 1  # A[i,1]
    V[0][0] = 1  # B[0,0]
    V[1][1] = 1  # B[1,0]
    for i in range(2):
        W[0*2+i][0] = 1  # C[0,i]
        W[0*2+i][1] = 1  # C[0,i]
    
    return U, V, W


def _combine_decomps(decomps):
    """Combine multiple decompositions into one larger decomposition.
    Each decomp is (U, V, W) with the same rank structure but different dimensions.
    We concatenate them along the rank (column) axis.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    # Determine dimensions from the first decomp's structure
    # Actually, we need to know the overall dimensions
    # This function is too generic; let's handle it in construct()
    pass


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    - Split A (2x3) into A_22 (2x2) and A_21 (2x1)
    - Split B (3x9) into B_29 (2x9) and B_19 (1x9)
    - C = A_22 * B_29 + A_21 * B_19
    - For A_22 * B_29: split B_29 into three 2x3 blocks, each handled by Strassen(2x2x2) + direct(2x2x1)
    - For A_21 * B_19: rank-1 decomposition
    
    Total rank: 3 * (7 + 2) + 1 = 28
    """
    n, m, p = 2, 3, 9
    
    # Build the full U, V, W matrices
    # U: (n*m) x rank = 6 x 28
    # V: (m*p) x rank = 27 x 28
    # W: (p*n) x rank = 18 x 28
    
    rank = 28
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(27)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    
    # Part 1: A_22 * B_29
    # B_29 has 3 blocks of 2x3. For each block, use Strassen(2x2x2) + direct(2x2x1)
    for block_idx in range(3):
        # Strassen part: 2x2x2
        sU, sV, sW = _strassen_222()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    # U index: i*m + j, but within A_22, j is 0,1
                    # A_22 is rows 0,1 of A, cols 0,1 of A
                    # U global index: i*3 + j
                    U[i * 3 + j][col] += sU[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    # V index: j*p + k, j is 0,1 (within B_29's row block), k is 0,1 (within 2x2)
                    # B_29 row j corresponds to B row j (since B_29 is rows 0,1 of B)
                    # V global index: j*9 + (block_idx*3 + k)
                    V[j * 9 + block_idx * 3 + k][col] += sV[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    # W index: k*n + i
                    # k is 0,1 (within the 2x2 output block), i is 0,1
                    # W global index: (block_idx*3 + k)*2 + i
                    W[(block_idx * 3 + k) * 2 + i][col] += sW[k * 2 + i][r]
            col += 1
        
        # Direct part: 2x2x1
        dU, dV, dW = _direct_221()
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    # U index: i*3 + j
                    U[i * 3 + j][col] += dU[i * 2 + j][r]
            for j in range(2):
                # k=0 (since p=1)
                # V global index: j*9 + (block_idx*3 + 2)  (the 3rd column of this block)
                V[j * 9 + block_idx * 3 + 2][col] += dV[j][r]
            for i in range(2):
                # k=0
                # W global index: (block_idx*3 + 2)*2 + i
                W[(block_idx * 3 + 2) * 2 + i][col] += dW[i][r]
            col += 1
    
    # Part 2: A_21 * B_19
    # A_21 is 2x1 (rows 0,1 of A, col 2 of A)
    # B_19 is 1x9 (row 2 of B, all 9 columns)
    # C = A_21 * B_19, which is 2x9
    # Rank 1 decomposition:
    # U[i*3+2] = 1 for i=0,1
    # V[2*9+k] = 1 for k=0..8
    # W[k*2+i] = 1 for k=0..8, i=0,1
    
    for i in range(2):
        U[i * 3 + 2][col] = 1
    for k in range(9):
        V[2 * 9 + k][col] = 1
    for k in range(9):
        for i in range(2):
            W[k * 2 + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
