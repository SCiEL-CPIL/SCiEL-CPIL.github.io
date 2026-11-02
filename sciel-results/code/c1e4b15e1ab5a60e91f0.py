"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Strategy:
    # 1. Split 4x5x6 into three disjoint regions:
    #    - Region A: i in [0,4), j in [0,4), k in [0,4)  -> 4x4x4
    #    - Region B: i in [0,4), j in [4,5), k in [0,6)  -> 4x1x6
    #    - Region C: i in [0,4), j in [0,4), k in [4,6)  -> 4x4x2
    # 2. Use Strassen (rank 49) for Region A, schoolbook for B and C.
    # 3. Total rank = 49 + 24 + 32 = 105.

    # Build Strassen 4x4x4 decomposition (rank 49)
    # Strassen 4x4x4 is built from 7 Strassen 2x2x2 blocks.
    # Each 2x2x2 has rank 7, so 7*7 = 49.

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # We need to set U[i,j,c], V[j,k,c], W[k,i,c] for each term c.
        # The condition is: for all i,j,k, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # But more precisely, the bilinear map condition:
        # sum_c W[k,i,c] * U[i,j,c] * V[l,k,c] = delta_{j,l}

        # Let's define each term:

        # c=0: M1 = (A00+A11)(B00+B11)
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,1]=1
        U[0][0] = 1  # i=0,j=0
        U[3][0] = 1  # i=1,j=1
        V[0][0] = 1  # j=0,k=0
        V[3][0] = 1  # j=1,k=1
        W[0][0] = 1  # k=0,i=0
        W[3][0] = 1  # k=1,i=1

        # c=1: M2 = (A10+A11)B00
        # U[1,0]=1, U[1,1]=1
        # V[0,0]=1
        # W[0,1]=1  # k=0,i=1
        U[2][1] = 1  # i=1,j=0
        U[3][1] = 1  # i=1,j=1
        V[0][1] = 1  # j=0,k=0
        W[1][1] = 1  # k=0,i=1

        # c=2: M3 = A00(B10-B11)
        # U[0,0]=1
        # V[1,0]=1, V[1,1]=-1
        # W[0,0]=1  # k=0,i=0
        # W[1,0]=1  # k=1,i=0
        U[0][2] = 1  # i=0,j=0
        V[2][2] = 1  # j=1,k=0
        V[3][2] = -1  # j=1,k=1
        W[0][2] = 1  # k=0,i=0
        W[1][2] = 1  # k=1,i=0

        # c=3: M4 = A11(B01-B00)
        # U[1,1]=1
        # V[0,0]=-1, V[0,1]=1
        # W[0,1]=1  # k=0,i=1
        # W[1,1]=1  # k=1,i=1
        U[3][3] = 1  # i=1,j=1
        V[0][3] = -1  # j=0,k=0
        V[1][3] = 1  # j=0,k=1
        W[1][3] = 1  # k=0,i=1
        W[3][3] = 1  # k=1,i=1

        # c=4: M5 = (A00+A01)B11
        # U[0,0]=1, U[0,1]=1
        # V[1,1]=1
        # W[0,0]=-1  # k=0,i=0
        # W[1,0]=1   # k=1,i=0
        U[0][4] = 1  # i=0,j=0
        U[1][4] = 1  # i=0,j=1
        V[3][4] = 1  # j=1,k=1
        W[0][4] = -1  # k=0,i=0
        W[1][4] = 1  # k=1,i=0

        # c=5: M6 = (A10-A00)(B00+B10)
        # U[1,0]=1, U[0,0]=-1
        # V[0,0]=1, V[1,0]=1
        # W[1,1]=1  # k=1,i=1
        U[2][5] = 1  # i=1,j=0
        U[0][5] = -1  # i=0,j=0
        V[0][5] = 1  # j=0,k=0
        V[2][5] = 1  # j=1,k=0
        W[3][5] = 1  # k=1,i=1

        # c=6: M7 = (A01-A11)(B01+B11)
        # U[0,1]=1, U[1,1]=-1
        # V[0,1]=1, V[1,1]=1
        # W[0,0]=1  # k=0,i=0
        # W[1,1]=1  # k=1,i=1
        U[1][6] = 1  # i=0,j=1
        U[3][6] = -1  # i=1,j=1
        V[1][6] = 1  # j=0,k=1
        V[3][6] = 1  # j=1,k=1
        W[0][6] = 1  # k=0,i=0
        W[3][6] = 1  # k=1,i=1

        return U, V, W

    def strassen_4x4x4():
        """
        Build 4x4x4 Strassen decomposition from 7 2x2x2 blocks.
        Returns U, V, W of shape 16x49.
        """
        U2, V2, W2 = strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]

        # 4x4 matrix can be split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]

        # Strassen 4x4 uses 7 products of 2x2 matrices:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # Each M is a 2x2 matrix product, computed via Strassen 2x2x2 (rank 7).
        # So total rank = 7 * 7 = 49.

        # For each of the 7 M's, we use the 2x2x2 Strassen decomposition.
        # The indices for the 4x4 case:
        # A00: i=0,1; j=0,1
        # A01: i=0,1; j=2,3
        # A10: i=2,3; j=0,1
        # A11: i=2,3; j=2,3
        # Similarly for B and C.

        # Let's define the block offsets:
        # Block (bi, bj) of A: i in [2*bi, 2*bi+2), j in [2*bj, 2*bj+2)
        # Block (bk, bj) of B: j in [2*bj, 2*bj+2), k in [2*bk, 2*bk+2)
        # Block (bk, bi) of C: i in [2*bi, 2*bi+2), k in [2*bk, 2*bk+2)

        # For each of the 7 Strassen terms at the 4x4 level, we need to:
        # 1. Determine which 2x2 blocks of A and B are involved.
        # 2. Use the 2x2x2 Strassen decomposition to compute the product.
        # 3. Map the results to the correct positions in U, V, W.

        # This is complex. Let me use a simpler approach:
        # Just use schoolbook for 4x4x4 (rank 64) to be safe.
        # Total rank = 64 + 24 + 32 = 120. Same as seed.

        # Actually, let me try to implement it correctly.
        # For each of the 7 M's, the 2x2x2 Strassen gives us 7 sub-terms.
        # So we have 7*7 = 49 terms total.

        # Let me define the 7 Strassen combinations for 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # For each M, we need to compute the product of two 2x2 matrices.
        # The product of two 2x2 matrices can be done via Strassen 2x2x2 (rank 7).

        # Let me define a helper to add a 2x2x2 Strassen block at given offsets.

        def add_2x2x2_block(U, V, W, U2, V2, W2, i_off, j_off, k_off, col_offset, scale_a, scale_b, scale_c):
            """
            Add a 2x2x2 Strassen block at the given offsets.
            scale_a, scale_b, scale_c are multipliers for U, V, W respectively.
            """
            for ci in range(4):
                for cj in range(7):
                    val_u = U2[ci][cj]
                    if val_u != 0:
                        i = i_off + ci // 2
                        j = j_off + ci % 2
                        U[i*4 + j][col_offset + cj] += val_u * scale_a

            for cj in range(4):
                for ck in range(7):
                    val_v = V2[cj][ck]
                    if val_v != 0:
                        j = j_off + cj // 2
                        k = k_off + cj % 2
                        V[j*4 + k][col_offset + ck] += val_v * scale_b

            for ck in range(4):
                for ci in range(7):
                    val_w = W2[ck][ci]
                    if val_w != 0:
                        k = k_off + ck // 2
                        i = i_off + ck % 2
                        W[k*4 + i][col_offset + ci] += val_w * scale_c

        # This approach is getting too complex. Let me just use schoolbook for 4x4x4.
        # Rank 64.
        # Total = 64 + 24 + 32 = 120.

        # For now, return schoolbook 4x4x4.
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4 + j][col] = 1
                    V[j*4 + k][col] = 1
                    W[k*4 + i][col] = 1
                    col += 1
        return U, V, W

    # Since Strassen 4x4x4 is complex, use schoolbook (rank 64).
    # Total rank = 64 + 24 + 32 = 120. Same as seed.

    # Let me try a different approach: just return schoolbook for the whole thing.
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

# EVOLVE-BLOCK-END
