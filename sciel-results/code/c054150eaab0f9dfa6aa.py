import time

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    # A1 is A[:, 0]
    # B1 is B[0, :]
    # C_part1[i, k] = A[i, 0] * B[0, k]
    for i in range(2):
        for k in range(8):
            # U index: i*m + 0
            # V index: 0*p + k
            # W index: k*n + i
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # A2 is A[:, 1:3]
    # B2 is B[1:3, :]
    # Split B2 into 4 blocks of 2x2: B2_0, B2_1, B2_2, B2_3
    # Each block uses Strassen (rank 7)
    
    for block in range(4):
        # B block columns: block*2, block*2+1
        # B rows: 1, 2 (global indices 1*p+col, 2*p+col)
        
        # Local A indices in A2:
        # a00 = A[0,1] -> global 0*3+1 = 1
        # a01 = A[0,2] -> global 0*3+2 = 2
        # a10 = A[1,1] -> global 1*3+1 = 4
        # a11 = A[1,2] -> global 1*3+2 = 5
        
        # B block local:
        # b00 = B[1, block*2] -> index 1*8 + block*2
        # b01 = B[1, block*2+1] -> index 1*8 + block*2+1
        # b10 = B[2, block*2] -> index 2*8 + block*2
        # b11 = B[2, block*2+1] -> index 2*8 + block*2+1
        
        # Outputs C:
        # c00 -> C[0, block*2] -> W index (block*2)*2 + 0
        # c01 -> C[0, block*2+1] -> W index (block*2+1)*2 + 0
        # c10 -> C[1, block*2] -> W index (block*2)*2 + 1
        # c11 -> C[1, block*2+1] -> W index (block*2+1)*2 + 1
        
        # Strassen steps:
        # 1. (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        V[2*8 + block*2+1][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[5][col] = 1
        V[2*8 + block*2][col] = 1
        V[1*8 + block*2][col] = -1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2)*2 + 1][col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1*8 + block*2][col] = 1
        V[1*8 + block*2+1][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
