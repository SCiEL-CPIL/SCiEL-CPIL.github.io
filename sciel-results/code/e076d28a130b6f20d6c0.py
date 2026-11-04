"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # 6x6x6 matrix multiplication via Strassen's algorithm on 3x3 blocks of 2x2.
    # Each 2x2x2 block uses rank 7 (Strassen). 3x3 = 9 block positions, but we need
    # cross-block terms too. Actually, for 6x6 = 3x3 blocks of 2x2, the standard
    # block matrix multiplication has 27 (bi,bj,bk) triples, each requiring a 2x2 
    # multiplication. Using Strassen per 2x2 block: 27 * 7 = 189 rank.
    #
    # However, we can do better by using Strassen at the block level too.
    # 6x6 = 3x3 blocks. We can apply Strassen to the 3x3 block structure by 
    # grouping blocks. But 3 is not a power of 2, so direct Strassen doesn't apply.
    #
    # Alternative: use the known rank-189 decomposition (Strassen per 2x2 sub-block).
    # The known optimal rank for 6x6 is 189 (or possibly lower, but 189 is the 
    # best known simple construction).
    
    n, m, p = DIMENSIONS
    # We'll use the 27-block Strassen approach: 27 * 7 = 189 rank terms.
    # Each term corresponds to a (bi, bj, bk) block triple and a Strassen term s.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A12)(B22)
    # M3 = (A21+A22)(B11)
    # M4 = (A22)(B22-B11)
    # M5 = (A11+A22)(B21-B12)
    # M6 = (A11-A21)(B12+B22)
    # M7 = (A22-A12)(B11+B21)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M5
    # C21 = M3 + M6
    # C22 = M1 - M2 + M4 + M5
    #
    # For each block position (bi, bj, bk), we have:
    #   A_block = A[2bi:2bi+2, 2bj:2bj+2]
    #   B_block = B[2bj:2bj+2, 2bk:2bk+2]
    #   C_block = C[2bk:2bk+2, 2bi:2bi+2]
    #
    # The Strassen terms give:
    #   s=0: U entries from A11, V entries from B11+B22
    #   s=1: U entries from A11+A12, V entries from B22
    #   s=2: U entries from A21+A22, V entries from B11
    #   s=3: U entries from A22, V entries from B22-B11
    #   s=4: U entries from A11+A22, V entries from B21-B12
    #   s=5: U entries from A11-A21, V entries from B12+B22
    #   s=6: U entries from A22-A12, V entries from B11+B21
    
    # Define the Strassen patterns for U, V, W for each of the 7 terms.
    # U has shape 4x1 for a 2x2 A, V has shape 4x1 for a 2x2 B, W has shape 4x1 for a 2x2 C.
    # We index U[row, col] where row = 2*ai + aj (ai, aj in {0,1})
    # V[row, col] where row = 2*aj + ak
    # W[row, col] where row = 2*ak + ai
    
    strassen_patterns = [
        # s=0: M1 = A11 * (B11+B22), C += M1 at C11, C22
        {
            'u': {(0,0): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # s=1: M2 = (A11+A12) * B22, C += M2 at C12, C22 (with -M2)
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,1): 1, (1,1): -1}
        },
        # s=2: M3 = (A21+A22) * B11, C += M3 at C21
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1}
        },
        # s=3: M4 = A22 * (B22-B11), C += M4 at C11, C22
        {
            'u': {(1,1): 1},
            'v': {(1,1): 1, (0,0): -1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # s=4: M5 = (A11+A22) * (B21-B12), C += M5 at C12, C21 (with -M5), C22
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(1,0): 1, (0,1): -1},
            'w': {(0,1): -1, (1,0): 1, (1,1): 1}
        },
        # s=5: M6 = (A11-A21) * (B12+B22), C += M6 at C21
        {
            'u': {(0,0): 1, (1,0): -1},
            'v': {(0,1): 1, (1,1): 1},
            'w': {(1,0): 1}
        },
        # s=6: M7 = (A22-A12) * (B11+B21), C += M7 at C11
        {
            'u': {(1,1): 1, (0,1): -1},
            'v': {(0,0): 1, (1,0): 1},
            'w': {(0,0): 1}
        },
    ]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for s in range(7):
                    pat = strassen_patterns[s]
                    for (ai, aj), u_val in pat['u'].items():
                        row_u = 2 * (2*bi + ai) + (2*bj + aj)
                        # Wait, U index is i*m + j where i is row of A, j is col of A.
                        # A is 6x6, so i in 0..5, j in 0..5.
                        # Block (bi, bj) means rows 2bi, 2bi+1 and cols 2bj, 2bj+1.
                        # So for ai, aj in {0,1}: i = 2*bi + ai, j = 2*bj + aj
                        row_u = (2*bi + ai) * m + (2*bj + aj)
                        U[row_u][col] = u_val
                    for (aj, ak), v_val in pat['v'].items():
                        # V index is j*p + k. j = 2*bj + aj, k = 2*bk + ak
                        row_v = (2*bj + aj) * p + (2*bk + ak)
                        V[row_v][col] = v_val
                    for (ak, ai), w_val in pat['w'].items():
                        # W index is k*n + i. k = 2*bk + ak, i = 2*bi + ai
                        row_w = (2*bk + ak) * n + (2*bi + ai)
                        W[row_w][col] = w_val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
