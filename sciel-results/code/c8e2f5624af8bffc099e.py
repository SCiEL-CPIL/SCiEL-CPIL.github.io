"""Matrix multiplication 2,4,6 - Strassen-based decomposition, rank 42."""
DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    # Split m=4 into two 2-blocks, p=6 into three 2-blocks
    # For each (m_block, p_block) pair, use Strassen 2x2x2 (rank 7)
    # Total rank = 2 * 3 * 7 = 42
    
    U = [[0] * 42 for _ in range(n * m)]  # 8 x 42
    V = [[0] * 42 for _ in range(m * p)]  # 24 x 42
    W = [[0] * 42 for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2: C = A(2x2) * B(2x2), trace convention W[k*2+i, r]
    # A indices: (i,j) where i=row(0,1), j=col(0,1) -> U index i*2+j
    # B indices: (j,k) where j=row(0,1), k=col(0,1) -> V index j*2+k
    # C indices: (k,i) where k=col(0,1), i=row(0,1) -> W index k*2+i
    #
    # Strassen factors (7 multiplications):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In trace convention W[k*2+i]:
    # W[0*2+0] = C11 (k=0,i=0)
    # W[0*2+1] = C21 (k=0,i=1)  -- wait, let me recheck
    #
    # Actually W[k*n+i] where n=2: W[k*2+i]
    # k is the column index of C (0,1), i is the row index of C (0,1)
    # So W[0*2+0]=C[0,0], W[0*2+1]=C[1,0], W[1*2+0]=C[0,1], W[1*2+1]=C[1,1]
    #
    # C[0,0]=C11, C[1,0]=C21, C[0,1]=C12, C[1,1]=C22
    # W[0]=C11, W[1]=C21, W[2]=C12, W[3]=C22
    
    strassen_U = [
        [ 1,  0,  1,  0,  1,  0,  0],  # A11: appears in M1,M3,M5,M6
        [ 0,  1,  0,  0,  0,  1,  0],  # A21: appears in M2,M6
        [ 0,  0,  0,  1,  0,  0,  1],  # A12: appears in M5,M7
        [ 1,  1,  0,  0,  0,  0,  0],  # A22: appears in M1,M2,M4,M7
    ]
    # Wait, let me redo this carefully.
    # A is 2x2: A[i][j], i=row, j=col
    # A11 = A[0][0], A12 = A[0][1], A21 = A[1][0], A22 = A[1][1]
    # U index for A[i][j] in a 2x2 block: i*2+j
    # So U[0]=A11, U[1]=A12, U[2]=A21, U[3]=A22
    
    # M1 = (A11+A22)(B11+B22): U has A11=+1, A22=+1; V has B11=+1, B22=+1
    # M2 = (A21+A22)B11: U has A21=+1, A22=+1; V has B11=+1
    # M3 = A11(B21-B22): U has A11=+1; V has B21=+1, B22=-1
    # M4 = A22(B12-B11): U has A22=+1; V has B12=+1, B11=-1
    # M5 = (A11+A12)B22: U has A11=+1, A12=+1; V has B22=+1
    # M6 = (A21-A11)(B11+B12): U has A21=+1, A11=-1; V has B11=+1, B12=+1
    # M7 = (A12-A22)(B21+B22): U has A12=+1, A22=-1; V has B21=+1, B22=+1
    
    # B is 2x2: B[j][k], j=row, k=col
    # B11 = B[0][0], B12 = B[0][1], B21 = B[1][0], B22 = B[1][1]
    # V index for B[j][k] in a 2x2 block: j*2+k
    # So V[0]=B11, V[1]=B12, V[2]=B21, V[3]=B22
    
    strassen_U = [
        [ 1,  0,  0,  1,  1, -1,  0],  # A11 (U[0])
        [ 0,  1,  0,  0,  1,  0,  1],  # A12 (U[1])
        [ 0,  0,  1,  0,  0,  1,  0],  # A21 (U[2])
        [ 1,  0,  0,  1,  0,  0, -1],  # A22 (U[3])
    ]
    strassen_V = [
        [ 1,  1,  0, -1,  0,  1,  0],  # B11 (V[0])
        [ 0,  0,  0,  1,  0,  1,  0],  # B12 (V[1])
        [ 0,  0,  1,  0,  0,  0,  1],  # B21 (V[2])
        [ 1,  0, -1,  0,  1,  0,  1],  # B22 (V[3])
    ]
    # W for 2x2x2, trace convention W[k*2+i]:
    # W[0]=C[0][0]=C11 = M1 + M4 - M5 + M7
    # W[1]=C[1][0]=C21 = M2 + M4
    # W[2]=C[0][1]=C12 = M3 + M5
    # W[3]=C[1][1]=C22 = M1 - M2 + M3 + M6
    strassen_W = [
        [ 1,  0,  0,  1, -1,  0,  1],  # W[0] = C11
        [ 0,  1,  0,  1,  0,  0,  0],  # W[1] = C21
        [ 0,  0,  1,  0,  1,  0,  0],  # W[2] = C12
        [ 1, -1, 1,  0,  0,  1,  0],  # W[3] = C22
    ]
    
    # Now fill in the full 2x4x6 decomposition
    # For each m_block (0 or 1) and p_block (0, 1, or 2):
    #   A indices: rows 0,1; cols m_block*2, m_block*2+1 -> U index i*4 + (m_block*2 + j)
    #   B indices: rows m_block*2, m_block*2+1; cols p_block*2, p_block*2+1 -> V index (m_block*2+j)*6 + (p_block*2+k)
    #   C indices: rows 0,1; cols p_block*2, p_block*2+1 -> W index (p_block*2+k)*2 + i
    
    rank = 42
    col = 0
    for m_block in range(2):
        for p_block in range(3):
            for r in range(7):
                # U: 4 entries (A is 2x2 within this block)
                for i in range(2):
                    for j in range(2):
                        u_idx = i * 4 + (m_block * 2 + j)
                        U[u_idx][col] = strassen_U[i * 2 + j][r]
                
                # V: 4 entries (B is 2x2 within this block)
                for j in range(2):
                    for k in range(2):
                        v_idx = (m_block * 2 + j) * 6 + (p_block * 2 + k)
                        V[v_idx][col] = strassen_V[j * 2 + k][r]
                
                # W: 4 entries (C is 2x2 within this block), trace convention W[k*n+i]
                for k in range(2):
                    for i in range(2):
                        w_idx = (p_block * 2 + k) * 2 + i
                        W[w_idx][col] = strassen_W[k * 2 + i][r]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
