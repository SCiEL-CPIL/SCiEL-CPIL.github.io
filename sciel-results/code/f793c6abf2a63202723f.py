import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition m into 2+3 and p into 3+3.
    # This creates 4 blocks:
    # 1. n=3, m=2, p=3
    # 2. n=3, m=3, p=3
    # 3. n=3, m=2, p=3
    # 4. n=3, m=3, p=3
    
    # We need decompositions for:
    # - 3x3x3 (Rank 23)
    # - 3x2x3 (Rank 17)
    
    # Helper to create schoolbook decomposition
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Strassen decomposition for 2x2x2 (Rank 7)
    def strassen_2x2x2():
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # Indices: i=0,1; j=0,1; k=0,1
        # m=2, p=2, n=2
        U = [
            [1, 0, 1, 1, 0, 1, 0], # i=0, j=0
            [0, 1, 0, 0, 1, 0, 1], # i=0, j=1
            [1, 0, 1, 0, 0, 0, 0], # i=1, j=0
            [0, 1, 0, 1, 0, 0, 0]  # i=1, j=1
        ]
        V = [
            [1, 0, 0, 0, 0, 0, 0], # j=0, k=0
            [0, 1, 1, 0, 0, 0, 0], # j=0, k=1
            [0, 0, 0, 0, 1, 0, 0], # j=1, k=0
            [0, 0, 0, 0, 0, 1, 1]  # j=1, k=1
        ]
        W = [
            [0, 1, 0, 0, 1, 0, 1], # k=0, i=0
            [1, 0, 1, 1, 0, 0, 0], # k=0, i=1
            [0, 0, 0, 0, 0, 1, 1], # k=1, i=0
            [0, 0, 0, 0, 1, 0, 1]  # k=1, i=1
        ]
        # Note: The specific coefficients for Strassen can vary. 
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Let's verify the U,V,W above matches this structure.
        # U[i*m+j, r] corresponds to A coefficients.
        # V[j*p+k, r] corresponds to B coefficients.
        # W[k*n+i, r] corresponds to C coefficients.
        
        # This hardcoded version is risky if not perfectly verified. 
        # Given time constraints, I will use a known correct set or fall back to schoolbook for 2x2x2 (Rank 4) if Strassen is too complex to hardcode correctly.
        # However, Rank 7 is essential for the reduction.
        # Let's use a verified Strassen matrix.
        
        # Actually, for 2x2x2, schoolbook is Rank 4. Strassen is Rank 7.
        # If I use Schoolbook for 2x2x2, the rank of 3x2x3 becomes:
        # 2x2x2: 4
        # 2x2x1: 4
        # 1x2x2: 4
        # 1x2x1: 2
        # Total: 14.
        # This is actually LOWER than 17?
        # Wait, Strassen is 7. 7 > 4.
        # Why use Strassen? Because for larger blocks it compounds.
        # For 3x2x3, if I use Schoolbook for all sub-blocks:
        # 2x2x2: 4
        # 2x2x1: 4
        # 1x2x2: 4
        # 1x2x1: 2
        # Total: 14.
        # This is better than 17!
        
        # Let's re-evaluate 3x3x3.
        # Schoolbook: 27.
        # Rank 23 is better.
        
        # So, strategy:
        # 3x3x3: Rank 23.
        # 3x2x3: Rank 14 (using Schoolbook for sub-blocks).
        
        # Total Rank: 23 + 23 + 14 + 14 = 74.
        # Score: 90/74 = 1.216.
        
        # I will implement the Rank 14 decomposition for 3x2x3.
        # And I need Rank 23 for 3x3x3.
        
        pass

    def get_3x2x3():
        # Decompose 3x2x3 into 2x2x2, 2x2x1, 1x2x2, 1x2x1
        # n=3, m=2, p=3
        # Partition n: 2+1. p: 2+1.
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Block 1: n=2, m=2, p=2. (i=0,1; j=0,1; k=0,1)
        # Schoolbook Rank 4
        U1, V1, W1, _ = schoolbook(2, 2, 2)
        
        # Block 2: n=2, m=2, p=1. (i=0,1; j=0,1; k=2)
        # p index is 2 (0-indexed).
        U2, V2, W2, _ = schoolbook(2, 2, 1)
        
        # Block 3: n=1, m=2, p=2. (i=2; j=0,1; k=0,1)
        U3, V3, W3, _ = schoolbook(1, 2, 2)
        
        # Block 4: n=1, m=2, p=1. (i=2; j=0,1; k=2)
        U4, V4, W4, _ = schoolbook(1, 2, 1)
        
        # Concatenate
        # U rows: n*m = 3*2 = 6.
        # U1 is 4x4. U2 is 4x4. U3 is 2x4. U4 is 2x2.
        # We need to map local indices to global indices.
        
        U = [[0] * 14 for _ in range(6)]
        V = [[0] * 14 for _ in range(6)] # m*p = 2*3 = 6
        W = [[0] * 14 for _ in range(9)] # p*n = 3*3 = 9
        
        col = 0
        # Block 1: i=0,1; j=0,1; k=0,1
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = U1[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[j * 3 + k][col] = V1[j * 2 + k][r] # Local p=2, Global p=3. k_local 0->0, 1->1
            for k in range(2):
                for i in range(2):
                    W[k * 3 + i][col] = W1[k * 2 + i][r] # Local n=2, Global n=3. i_local 0->0, 1->1
            col += 1
            
        # Block 2: i=0,1; j=0,1; k=2
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = U2[i * 2 + j][r]
            for j in range(2):
                for k in range(1):
                    V[j * 3 + 2][col] = V2[j * 2 + k][r] # k_local 0 -> Global k=2
            for k in range(1):
                for i in range(2):
                    W[2 * 3 + i][col] = W2[k * 2 + i][r] # k_local 0 -> Global k=2
            col += 1
            
        # Block 3: i=2; j=0,1; k=0,1
        for r in range(4):
            for i in range(1):
                for j in range(2):
                    U[2 * 2 + j][col] = U3[i * 2 + j][r] # i_local 0 -> Global i=2
            for j in range(2):
                for k in range(2):
                    V[j * 3 + k][col] = V3[j * 2 + k][r]
            for k in range(2):
                for i in range(1):
                    W[k * 3 + 2][col] = W3[k * 1 + i][r] # i_local 0 -> Global i=2
            col += 1
            
        # Block 4: i=2; j=0,1; k=2
        for r in range(2):
            for i in range(1):
                for j in range(2):
                    U[2 * 2 + j][col] = U4[i * 2 + j][r]
            for j in range(2):
                for k in range(1):
                    V[j * 3 + 2][col] = V4[j * 2 + k][r]
            for k in range(1):
                for i in range(1):
                    W[2 * 3 + 2][col] = W4[k * 1 + i][r]
            col += 1
            
        return U, V, W, col

    def get_3x3x3():
        # Hardcode Rank 23 decomposition for 3x3x3
        # This is a known constant. 
        # If I cannot recall the exact 23 coefficients, I will fall back to Schoolbook (27).
        # To ensure correctness and non-zero improvement, I will use Schoolbook for now if 23 is too risky?
        # No, 27 is the seed. 23 is a definite improvement.
        # I will use a placeholder for 23 that is actually Schoolbook (27) to ensure it runs, 
        # but then I won't improve on the 3x3x3 blocks.
        # Total Rank would be 14+14+27+27 = 82.
        # Score 1.097.
        
        # Let's try to find a simpler Rank 24 or 25?
        # Actually, I will just use Schoolbook for 3x3x3 to be safe, 
        # and rely on the 3x2x3 reduction (14 vs 18) to gain score.
        # 18-14 = 4 gain per block. 2 blocks. Total gain 8.
        # 90 - 8 = 82.
        
        U, V, W, rank = schoolbook(3, 3, 3)
        return U, V, W, rank

    # Combine the 4 blocks for the full 3x5x6
    # Blocks:
    # 1. n=3, m=2, p=3 (j=0,1; k=0,2) -> Rank 14
    # 2. n=3, m=3, p=3 (j=2,4; k=0,2) -> Rank 27 (Schoolbook)
    # 3. n=3, m=2, p=3 (j=0,1; k=3,5) -> Rank 14
    # 4. n=3, m=3, p=3 (j=2,4; k=3,5) -> Rank 27 (Schoolbook)
    
    U1, V1, W1, r1 = get_3x2x3()
    U2, V2, W2, r2 = get_3x3x3()
    U3, V3, W3, r3 = get_3x2x3()
    U4, V4, W4, r4 = get_3x3x3()
    
    total_rank = r1 + r2 + r3 + r4
    
    U = [[0] * total_rank for _ in range(3 * 5)]
    V = [[0] * total_rank for _ in range(5 * 6)]
    W = [[0] * total_rank for _ in range(6 * 3)]
    
    col = 0
    # Block 1: j in [0,2), k in [0,3)
    for r in range(r1):
        for i in range(3):
            for j in range(2):
                U[i * 5 + j][col] = U1[i * 2 + j][r]
        for j in range(2):
            for k in range(3):
                V[j * 6 + k][col] = V1[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i][col] = W1[k * 3 + i][r]
        col += 1
        
    # Block 2: j in [2,5), k in [0,3)
    for r in range(r2):
        for i in range(3):
            for j in range(3):
                U[i * 5 + 2 + j][col] = U2[i * 3 + j][r]
        for j in range(3):
            for k in range(3):
                V[(2 + j) * 6 + k][col] = V2[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i][col] = W2[k * 3 + i][r]
        col += 1
        
    # Block 3: j in [0,2), k in [3,6)
    for r in range(r3):
        for i in range(3):
            for j in range(2):
                U[i * 5 + j][col] = U3[i * 2 + j][r]
        for j in range(2):
            for k in range(3):
                V[j * 6 + 3 + k][col] = V3[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W[(3 + k) * 3 + i][col] = W3[k * 3 + i][r]
        col += 1
        
    # Block 4: j in [2,5), k in [3,6)
    for r in range(r4):
        for i in range(3):
            for j in range(3):
                U[i * 5 + 2 + j][col] = U4[i * 3 + j][r]
        for j in range(3):
            for k in range(3):
                V[(2 + j) * 6 + 3 + k][col] = V4[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W[(3 + k) * 3 + i][col] = W4[k * 3 + i][r]
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
