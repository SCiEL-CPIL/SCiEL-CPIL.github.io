"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of size 3x3.
    # Total rank = 7 * (3*3*3) = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define coefficients for Strassen's 7 multiplications
    # coeffA[s][bi][bj] = coefficient for A block (bi, bj) in X_s
    # coeffB[s][bi][bj] = coefficient for B block (bi, bj) in Y_s
    # coeffC[s][bi][bj] = coefficient for Z_s in C block (bi, bj)
    
    coeffA = [[[[0, 0], [0, 0]] for _ in range(2)] for _ in range(7)]
    coeffB = [[[[0, 0], [0, 0]] for _ in range(2)] for _ in range(7)]
    coeffC = [[[[0, 0], [0, 0]] for _ in range(2)] for _ in range(7)]
    
    # s=0: M1 = (A00 + A11)(B00 + B11)
    coeffA[0][0][0] = 1
    coeffA[0][1][1] = 1
    coeffB[0][0][0] = 1
    coeffB[0][1][1] = 1
    coeffC[0][0][0] = 1
    coeffC[0][1][1] = 1
    
    # s=1: M2 = (A10 + A11)B00
    coeffA[1][1][0] = 1
    coeffA[1][1][1] = 1
    coeffB[1][0][0] = 1
    coeffC[1][1][0] = 1
    coeffC[1][1][1] = -1
    
    # s=2: M3 = A00(B01 - B11)
    coeffA[2][0][0] = 1
    coeffB[2][0][1] = 1
    coeffB[2][1][1] = -1
    coeffC[2][0][1] = 1
    coeffC[2][1][1] = 1
    
    # s=3: M4 = A11(B10 - B00)
    coeffA[3][1][1] = 1
    coeffB[3][1][0] = 1
    coeffB[3][0][0] = -1
    coeffC[3][0][0] = 1
    coeffC[3][1][0] = 1
    
    # s=4: M5 = (A00 + A01)B11
    coeffA[4][0][0] = 1
    coeffA[4][0][1] = 1
    coeffB[4][1][1] = 1
    coeffC[4][0][0] = -1
    coeffC[4][0][1] = 1
    
    # s=5: M6 = (A10 - A00)(B00 + B01)
    coeffA[5][1][0] = 1
    coeffA[5][0][0] = -1
    coeffB[5][0][0] = 1
    coeffB[5][0][1] = 1
    coeffC[5][1][1] = 1
    
    # s=6: M7 = (A01 - A11)(B10 + B11)
    coeffA[6][0][1] = 1
    coeffA[6][1][1] = -1
    coeffB[6][1][0] = 1
    coeffB[6][1][1] = 1
    coeffC[6][0][0] = 1
    
    block_size = 3
    for s in range(7):
        for r in range(block_size):
            for c in range(block_size):
                for k in range(block_size):
                    col = s * (block_size ** 3) + (r * block_size + c) * block_size + k
                    
                    # Update U (Input A)
                    # X_s[r, c] = sum_{bi, bj} coeffA[s][bi][bj] * A[3*bi+r, 3*bj+c]
                    for bi in range(2):
                        for bj in range(2):
                            val = coeffA[s][bi][bj]
                            if val != 0:
                                I = 3 * bi + r
                                J = 3 * bj + c
                                U[I * m + J][col] += val
                                
                    # Update V (Input B)
                    # Y_s[c, k] = sum_{bi, bj} coeffB[s][bi][bj] * B[3*bi+c, 3*bj+k]
                    for bi in range(2):
                        for bj in range(2):
                            val = coeffB[s][bi][bj]
                            if val != 0:
                                I = 3 * bi + c
                                J = 3 * bj + k
                                V[I * p + J][col] += val
                                
                    # Update W (Output C)
                    # C_block(bi, bj)[r, k] += coeffC[s][bi][bj] * Z_s[r, k]
                    # W index is k * n + i (col * n + row)
                    for bi in range(2):
                        for bj in range(2):
                            val = coeffC[s][bi][bj]
                            if val != 0:
                                I = 3 * bi + r
                                J = 3 * bj + k
                                W[J * n + I][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
