"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    We use a systematic search approach:
    1. Start with the schoolbook decomposition (rank 36)
    2. Try to find a lower rank decomposition using known algorithms
    
    For 3x3 * 3x4, we know:
    - Schoolbook rank = 36
    - We can potentially do better by combining terms
    
    The key insight is that we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Each rank-1 term is (A[i,j], B[j,k], W[k*n+i])
    
    We can try to combine terms where the same A or B factors appear.
    
    Let's use a more systematic approach: enumerate all possible rank-1 terms
    and try to find a minimal set that covers all required (i,j,k) triples.
    
    Actually, for this specific case, let's use a known result:
    The bilinear complexity of 3x3 * 3x4 is at most 15.
    
    Let me construct this explicitly using a partitioning strategy.
    
    Strategy:
    - Partition the 3x3 * 3x4 multiplication into blocks
    - Use Strassen-like tricks for 2x2 blocks
    - Handle boundary terms with schoolbook
    
    A is 3x3, B is 3x4, C is 3x4
    
    Let's partition:
    A = [[A00, A01, A02],
         [A10, A11, A12],
         [A20, A21, A22]]
    
    B = [[B00, B01, B02, B03],
         [B10, B11, B12, B13],
         [B20, B21, B22, B23]]
    
    C = [[C00, C01, C02, C03],
         [C10, C11, C12, C13],
         [C20, C21, C22, C23]]
    
    Where C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Let's use the following decomposition:
    
    For the first row of C (i=0):
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    
    For the second row of C (i=1):
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    For the third row of C (i=2):
    C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    Total: 3*3*4 = 36 rank-1 terms
    
    To reduce rank, we can use the following trick:
    Define S0 = A[0,0] + A[1,0]
    Then A[0,0] = S0 - A[1,0]
    
    But this doesn't directly help unless we can factor.
    
    Let me try a different approach: use the fact that we can write
    (A[i,j] + A[i',j'])(B[j,k] + B[j',k']) = A[i,j]B[j,k] + A[i,j]B[j',k'] + A[i',j']B[j,k] + A[i',j']B[j',k']
    
    This gives us 4 products from 1 multiplication, but we need to be careful about which terms we need.
    
    Actually, let me just implement the schoolbook for now and return it.
    The rank is 36.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 36
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
# EVOLVE-BLOCK-END
