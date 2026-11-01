"""Matrix multiplication 2x5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 120
    
    # We decompose 2x5x6 as:
    # - 2x1x6 block (j=0): schoolbook, rank 12
    # - 2x4x6 block (j=1,2,3,4): split into two 2x2x6 blocks
    #   Each 2x2x6 uses Strassen (rank 7 per 2x2x2, so 3*7=21 per 2x2x6)
    #   Total for 2x4x6: 42
    # Grand total: 12 + 42 = 54
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
        U: (i,j) indexed as [i*2+j], 4 entries
        V: (j,k) indexed as [j*2+k], 4 entries
        W: (k,i) indexed as [k*2+i], 4 entries
        """
        # Using standard Strassen formulas:
        # M1 = (A00+A11)(B00+B11), C00+=1, C11+=1
        # M2 = (A10+A11)B00, C10+=1
        # M3 = (A00+A01)B11, C01+=1
        # M4 = (A00-A11)(B10-B11), C00+=1, C11+=1
        # M5 = (A00+A01)B00, C00+=1, C01+=1
        # M6 = (A10-A00)(B00+B01), C11+=1
        # M7 = (A01-A11)(B10+B11), C00+=1, C01+=1
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1
        U[0][0] = 1; U[3][0] = 1  # A00 + A11
        V[0][0] = 1; V[3][0] = 1  # B00 + B11
        W[0][0] = 1; W[3][0] = 1  # C00 + C11
        
        # r=1: M2
        U[2][1] = 1; U[3][1] = 1  # A10 + A11
        V[0][1] = 1               # B00
        W[2][1] = 1               # C10
        
        # r=2: M3
        U[0][2] = 1; U[1][2] = 1  # A00 + A01
        V[3][2] = 1               # B11
        W[1][2] = 1               # C01
        
        # r=3: M4
        U[0][3] = 1; U[3][3] = -1  # A00 - A11
        V[2][3] = 1; V[3][3] = -1  # B10 - B11
        W[0][3] = 1; W[3][3] = 1  # C00 + C11
        
        # r=4: M5
        U[0][4] = 1; U[1][4] = 1  # A00 + A01
        V[0][4] = 1               # B00
        W[0][4] = 1; W[1][4] = 1  # C00 + C01
        
        # r=5: M6
        U[2][5] = 1; U[0][5] = -1  # A10 - A00
        V[0][5] = 1; V[1][5] = 1  # B00 + B01
        W[3][5] = 1               # C11
        
        # r=6: M7
        U[1][6] = 1; U[3][6] = -1  # A01 - A11
        V[2][6] = 1; V[3][6] = 1  # B10 + B11
        W[0][6] = 1; W[1][6] = 1  # C00 + C01
        
        return U, V, W
    
    # Verify Strassen decomposition
    def verify_strassen():
        U, V, W = strassen_2x2x2()
        # Reconstruct tensor: T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(7))
                    expected = 1 if True else 0  # All entries should be 1 for 2x2x2
                    if val != expected:
                        return False
        return True
    
    strassen_ok = verify_strassen()
    
    if not strassen_ok:
        # Fall back to full schoolbook
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
    
    # Build the full 2x5x6 decomposition
    # Block 1: 2x1x6 (j=0), schoolbook rank 12
    # Block 2: 2x4x6 (j=1,2,3,4), using Strassen
    #   Sub-block 2a: 2x2x6 (j=1,2), rank 21
    #   Sub-block 2b: 2x2x6 (j=3,4), rank 21
    
    total_rank = 12 + 21 + 21  # 54
    
    U = [[0] * total_rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 54
    
    col = 0
    
    # Block 1: 2x1x6, j=0, schoolbook
    for i in range(n):  # i=0,1
        for k in range(p):  # k=0..5
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: 2x2x6, j=1,2, using Strassen (3 instances for 6 columns)
    U_s, V_s, W_s = strassen_2x2x2()
    for k_block in range(3):  # 3 blocks of 2 columns each
        for r in range(7):
            # U: i in {0,1}, j in {1,2} -> U[i*5 + (1+j_local)]
            for i in range(2):
                for j_local in range(2):
                    j = 1 + j_local  # j=1 or 2
                    U[i * m + j][col] += U_s[i * 2 + j_local][r]
            # V: j in {1,2}, k in {k_block*2, k_block*2+1}
            for j_local in range(2):
                for k_local in range(2):
                    j = 1 + j_local
                    k = k_block * 2 + k_local
                    V[j * p + k][col] += V_s[j_local * 2 + k_local][r]
            # W: k in {k_block*2, k_block*2+1}, i in {0,1}
            for k_local in range(2):
                for i in range(2):
                    k = k_block * 2 + k_local
                    W[k * n + i][col] += W_s[k_local * 2 + i][r]
            col += 1
    
    # Block 2b: 2x2x6, j=3,4, using Strassen (3 instances for 6 columns)
    for k_block in range(3):  # 3 blocks of 2 columns each
        for r in range(7):
            for i in range(2):
                for j_local in range(2):
                    j = 3 + j_local  # j=3 or 4
                    U[i * m + j][col] += U_s[i * 2 + j_local][r]
            for j_local in range(2):
                for k_local in range(2):
                    j = 3 + j_local
                    k = k_block * 2 + k_local
                    V[j * p + k][col] += V_s[j_local * 2 + k_local][r]
            for k_local in range(2):
                for i in range(2):
                    k = k_block * 2 + k_local
                    W[k * n + i][col] += W_s[k_local * 2 + i][r]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
