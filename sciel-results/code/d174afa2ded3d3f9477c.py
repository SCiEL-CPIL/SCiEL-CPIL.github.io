import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    # Use Strassen decomposition for the core computation
    # For 4x4x4, we can use 2x2x2 Strassen recursively
    # Strassen 2x2x2: 7 multiplications instead of 8
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, each shape (4, 7)."""
        U = [
            [1, 0, 1, 0, 1, -1, 0],   # a: M1, M3, M5, M6
            [0, 0, 0, 0, 1, 0, 1],    # b: M5, M7
            [0, 1, 0, 0, 0, 1, 0],    # c: M2, M6
            [1, 1, 0, 1, 0, 0, -1],   # d: M1, M2, M4, M7
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0],   # e: M1, M2, M4, M6
            [0, 0, 1, 0, 0, 1, 0],    # f: M3, M6
            [0, 0, -1, 1, 0, 0, 1],   # g: M3, M4, M7
            [1, 0, 0, 0, 1, 0, 1],    # h: M1, M5, M7
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1],   # C00: M1, M4, M5, M7
            [0, 0, 1, 0, 1, 0, 0],    # C01: M3, M5
            [0, 1, 0, 1, 0, 0, 0],    # C10: M2, M4
            [1, -1, 1, 0, 0, 1, 0],   # C11: M1, M2, M3, M6
        ]
        return U, V, W
    
    # Build 4x4x4 decomposition using block Strassen
    # Treat 4x4x4 as 2x2x2 of 2x2 blocks
    def build_4x4x4():
        U2, V2, W2 = strassen_2x2x2()
        
        # For 4x4x4, we have 2x2 blocks of size 2x2
        # The full decomposition will have rank 7*4 = 28 (each scalar product expanded to 4)
        # But we need to be more careful about the block structure
        
        n, m, p = 4, 4, 4
        rank = 7 * 4  # 28
        U = [[0] * rank for _ in range(n * m)]  # 16 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 16 x 28
        
        # Block indices: for 2x2 blocks
        # A block (i,j) covers A[i*2:(i+1)*2, j*2:(j+1)*2]
        # In flattened: A[i*2*2 + j*2 + di + dj*2] = A[i*4 + j*2 + di + dj*2]
        
        # For each of the 7 Strassen products, we do a 2x2x2 multiplication
        for s in range(7):  # Strassen step
            for di in range(2):  # block row offset
                for dj in range(2):  # block col offset
                    for dk in range(2):  # block col offset for output
                    
                        # U: input A[i*4 + j*2 + di + dj*2]
                        a_idx = di + dj * 2
                        u_val = U2[di + dj * 2][s]
                        if u_val != 0:
                            for i_block in range(2):
                                for j_block in range(2):
                                    row = i_block * 4 + j_block * 2 + a_idx
                                    col = s * 4 + dk * 2 + 0  # need proper indexing
                                    # Actually, let me rethink the indexing
                                    pass
        
        # Let me use a cleaner approach with explicit expansion
        # 4x4x4 = 2x2x2 blocks, each 2x2x2
        # Total rank = 7 * 2 * 2 * 2 = 56? No, that's not right either.
        
        # Let me just use the known rank-49 or fall back to schoolbook
        # Actually, let's try a different strategy: use schoolbook for the full 4x4x6
        # and see if we can do better
        
        # For now, return schoolbook as fallback
        n, m, p = 4, 4, 6
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
        return U, V, W
    
    # Given complexity, use schoolbook decomposition
    n, m, p = 4, 4, 6
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
