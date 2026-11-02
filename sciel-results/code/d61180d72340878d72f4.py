import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 150s wall-clock limit, 5s margin
    n, m, p = DIMENSIONS
    
    # For multiplying (n x m) by (m x p) to get (n x p):
    # The tensor is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Output W follows trace convention W[k*n + i, r]
    #
    # Schoolbook: rank = n*m*p = 2*2*8 = 32
    # We can potentially do better using Strassen-like methods for the 2x2 part.
    # For 2x2 matrix multiplication, Strassen gives rank 7 instead of 4.
    # For (2x2) * (2x8), we can decompose the 2x8 matrix into four 2x2 blocks.
    # Then we have 4 block multiplications of 2x2 * 2x2, each of rank 7.
    # Total rank = 4 * 7 = 28.
    #
    # Let's construct the rank-28 decomposition.
    #
    # A is 2x2: A = [[a11, a12], [a21, a22]]
    # B is 2x8: B = [B0, B1, B2, B3] where each Bk is 2x2
    # C = A * B is 2x8: C = [C0, C1, C2, C3] where Ck = A * Bk
    #
    # For each block multiplication Ck = A * Bk, use Strassen's 7 products.
    #
    # Strassen for 2x2:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Indexing:
    # A indices: a11 -> U[0], a12 -> U[1], a21 -> U[2], a22 -> U[3]
    # Bk indices for block k: b11 -> V[k*4+0], b12 -> V[k*4+1], b21 -> V[k*4+2], b22 -> V[k*4+3]
    # Ck indices: c11 -> W[k*4+0], c12 -> W[k*4+1], c21 -> W[k*4+2], c22 -> W[k*4+3]
    #
    # For each block k (0..3), we have 7 Strassen products.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # For each block k, assign 7 columns
    # U rows: 0=a11, 1=a12, 2=a21, 3=a22
    # V rows for block k: k*4+0=b11, k*4+1=b12, k*4+2=b21, k*4+3=b22
    # W rows for block k: k*4+0=c11, k*4+1=c12, k*4+2=c21, k*4+3=c22
    
    for k in range(4):
        base = k * 7
        # M1 = (a11 + a22)(b11 + b22)
        U[0][base + 0] = 1
        U[3][base + 0] = 1
        V[k*4 + 0][base + 0] = 1
        V[k*4 + 3][base + 0] = 1
        W[k*4 + 0][base + 0] = 1   # c11 += M1
        W[k*4 + 3][base + 0] = 1   # c22 += M1
        
        # M2 = (a21 + a22)b11
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k*4 + 0][base + 1] = 1
        W[k*4 + 2][base + 1] = 1   # c21 += M2
        W[k*4 + 3][base + 1] = -1  # c22 -= M2
        
        # M3 = a11(b12 - b22)
        U[0][base + 2] = 1
        V[k*4 + 1][base + 2] = 1
        V[k*4 + 3][base + 2] = -1
        W[k*4 + 1][base + 2] = 1   # c12 += M3
        W[k*4 + 3][base + 2] = 1   # c22 += M3
        
        # M4 = a22(b21 - b11)
        U[3][base + 3] = 1
        V[k*4 + 2][base + 3] = 1
        V[k*4 + 0][base + 3] = -1
        W[k*4 + 0][base + 3] = 1   # c11 += M4
        W[k*4 + 2][base + 3] = 1   # c21 += M4
        
        # M5 = (a11 + a12)b22
        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[k*4 + 3][base + 4] = 1
        W[k*4 + 0][base + 4] = -1  # c11 -= M5
        W[k*4 + 1][base + 4] = 1   # c12 += M5
        
        # M6 = (a21 - a11)(b11 + b12)
        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k*4 + 0][base + 5] = 1
        V[k*4 + 1][base + 5] = 1
        W[k*4 + 3][base + 5] = 1   # c22 += M6
        
        # M7 = (a12 - a22)(b21 + b22)
        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[k*4 + 2][base + 6] = 1
        V[k*4 + 3][base + 6] = 1
        W[k*4 + 0][base + 6] = 1   # c11 += M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
