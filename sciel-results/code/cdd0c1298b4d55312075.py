"""Rewritten program for matrix multiplication 2x3 x 3x10 using Strassen block decomposition."""

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We split the middle dimension m=3 into two parts:
    # Part 1: j in {0, 1} (size 2x2 blocks, 5 blocks of 2x2 since p=10 -> 5 pairs)
    # Part 2: j = 2 (single column, schoolbook)
    
    # For Part 1: 5 independent 2x2x2 matrix multiplications, each solved by Strassen (rank 7)
    # For Part 2: 2x1x10 schoolbook (rank 20)
    # Total rank = 5 * 7 + 20 = 35 + 20 = 55
    
    total_rank = 55
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    # Strassen formulas for 2x2 matrix multiplication:
    # Given A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # Strassen's 7 multiplications:
    # M0 = (a00 + a11) * (b00 + b11)
    # M1 = (a10 + a11) * b00
    # M2 = a00 * (b01 - b11)
    # M3 = a11 * (b10 - b00)
    # M4 = (a00 + a01) * b11
    # M5 = (a10 - a00) * (b00 + b10)
    # M6 = (a01 - a11) * (b10 + b11)
    #
    # Results:
    # c00 = M0 + M3 - M5 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M4
    
    # Strassen coefficient tables (local 2x2 indices)
    # U coefficients: A-part for each of the 7 Strassen mults
    # Local A index: (i_local, j_local) -> i_local*2 + j_local for i_local,j_local in {0,1}
    # U_strassen[s][i_local*2 + j_local]
    U_strassen = [
        # s=0: M0 = (a00+a11)*(b00+b11)
        [1, 0, 0, 1],   # a00=1, a01=0, a10=0, a11=1
        # s=1: M1 = (a10+a11)*b00
        [0, 0, 1, 1],   # a00=0, a01=0, a10=1, a11=1
        # s=2: M2 = a00*(b01-b11)
        [1, 0, 0, 0],   # a00=1, a01=0, a10=0, a11=0
        # s=3: M3 = a11*(b10-b00)
        [0, 0, 0, 1],   # a00=0, a01=0, a10=0, a11=1
        # s=4: M4 = (a00+a01)*b11
        [1, 1, 0, 0],   # a00=1, a01=1, a10=0, a11=0
        # s=5: M5 = (a10-a00)*(b00+b10)
        [-1, 0, 1, 0],  # a00=-1, a01=0, a10=1, a11=0
        # s=6: M6 = (a01-a11)*(b10+b11)
        [0, 1, 0, -1],  # a00=0, a01=1, a10=0, a11=-1
    ]
    
    # V coefficients: B-part for each of the 7 Strassen mults
    # Local B index: (j_local, k_local) -> j_local*2 + k_local for j_local,k_local in {0,1}
    V_strassen = [
        # s=0: M0 = (a00+a11)*(b00+b11)
        [1, 0, 0, 1],   # b00=1, b01=0, b10=0, b11=1
        # s=1: M1 = (a10+a11)*b00
        [1, 0, 0, 0],   # b00=1, b01=0, b10=0, b11=0
        # s=2: M2 = a00*(b01-b11)
        [0, 1, 0, -1],  # b00=0, b01=1, b10=0, b11=-1
        # s=3: M3 = a11*(b10-b00)
        [-1, 0, 1, 0],  # b00=-1, b01=0, b10=1, b11=0
        # s=4: M4 = (a00+a01)*b11
        [0, 0, 0, 1],   # b00=0, b01=0, b10=0, b11=1
        # s=5: M5 = (a10-a00)*(b00+b10)
        [1, 0, 1, 0],   # b00=1, b01=0, b10=1, b11=0
        # s=6: M6 = (a01-a11)*(b10+b11)
        [0, 0, 1, 1],   # b00=0, b01=0, b10=1, b11=1
    ]
    
    # W coefficients: C-part for each of the 7 Strassen mults
    # Local C index: (k_local, i_local) -> k_local*2 + i_local for k_local,i_local in {0,1}
    # c00 = M0 + M3 - M5 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M4
    W_strassen = [
        # s=0: c00+=1, c11+=1
        [1, 0, 0, 0],   # c00=1 (k=0,i=0), c01=0, c10=0, c11=1... wait let me redo
    ]
    
    # Let me be more careful with W indices.
    # W[k*n + i, c] where k_local in {0,1}, i_local in {0,1}
    # Local index = k_local * 2 + i_local
    # c00: k_local=0, i_local=0 -> idx 0
    # c01: k_local=1, i_local=0 -> idx 2
    # c10: k_local=0, i_local=1 -> idx 1
    # c11: k_local=1, i_local=1 -> idx 3
    
    # c00 = M0 + M3 - M5 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M4
    
    W_strassen = [
        # s=0: c00+=1, c11+=1
        [1, 0, 0, 1],   # idx0(c00)=1, idx1(c10)=0, idx2(c01)=0, idx3(c11)=1
        # s=1: c10+=1, c11-=1
        [0, 1, 0, -1],  # idx0=0, idx1(c10)=1, idx2(c01)=0, idx3(c11)=-1
        # s=2: c01+=1, c11+=1
        [0, 0, 1, 1],   # idx0=0, idx1=0, idx2(c01)=1, idx3(c11)=1
        # s=3: c00+=1, c10+=1
        [1, 1, 0, 0],   # idx0(c00)=1, idx1(c10)=1, idx2=0, idx3=0
        # s=4: c01+=1, c11+=1
        [0, 0, 1, 1],   # idx0=0, idx1=0, idx2(c01)=1, idx3(c11)=1
        # s=5: c00-=1
        [-1, 0, 0, 0],  # idx0(c00)=-1, rest 0
        # s=6: c00+=1
        [1, 0, 0, 0],   # idx0(c00)=1, rest 0
    ]
    
    # Fill in U, V, W for Part 1 (j in {0,1}, 5 blocks of k pairs)
    for block in range(5):  # 5 blocks, each covering k=2*block, 2*block+1
        for s in range(7):  # 7 Strassen mults per block
            col = block * 7 + s  # column index in total_rank
            k_start = block * 2
            
            # Fill U: for each (i_local, j_local) in 2x2
            for i_local in range(2):
                for j_local in range(2):
                    coeff = U_strassen[s][i_local * 2 + j_local]
                    if coeff != 0:
                        U[i_local * 3 + j_local][col] += coeff  # global j = j_local
            
            # Fill V: for each (j_local, k_local) in 2x2
            for j_local in range(2):
                for k_local in range(2):
                    coeff = V_strassen[s][j_local * 2 + k_local]
                    if coeff != 0:
                        k_global = k_start + k_local
                        V[j_local * 10 + k_global][col] += coeff
            
            # Fill W: for each (k_local, i_local) in 2x2
            for k_local in range(2):
                for i_local in range(2):
                    coeff = W_strassen[s][k_local * 2 + i_local]
                    if coeff != 0:
                        k_global = k_start + k_local
                        W[k_global * 2 + i_local][col] += coeff
    
    # Part 2: j=2, schoolbook for 2x1x10
    # For each (i, k) with i in {0,1}, k in {0..9}: one term
    for i in range(2):
        for k in range(10):
            col = 35 + i * 10 + k  # column index
            U[i * 3 + 2][col] = 1
            V[2 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
