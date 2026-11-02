import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 64 decomposition
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(2): # 0, 1
        for j_block in range(2): # 0, 1
            for k_block in range(2): # 0, 1
                # C[i_block, k_block] += A[i_block, j_block] B[j_block, k_block]
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            U[i_block*4 + j_block*2 + j][col] = 1
                            V[(j_block*2 + j)*8 + (k_block*4 + k)][col] = 1
                            W[(k_block*4 + k)*4 + (i_block*2 + i)][col] = 1
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
