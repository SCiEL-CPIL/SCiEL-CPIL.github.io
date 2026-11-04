"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid indices.
    #
    # Split A's 4 columns into two pairs: L = cols 0,1 and R = cols 2,3.
    # Split B's 6 columns into three pairs: blocks of cols (0,1), (2,3), (4,5).
    # Each (A-part, B-col-block) pair gives a 2x2 * 2x2 multiplication (rank 7 via Strassen).
    # Total: 2 * 3 * 7 = 42.
    #
    # Strassen for C = A*B where A, B are 2x2:
    #   A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    #   M1 = (a00+a11)(b00+b11)
    #   M2 = (a10+a11)b00
    #   M3 = a00(b01-b11)
    #   M4 = a11(b10-b00)
    #   M5 = (a00+a01)b11
    #   M6 = (a10-a00)(b00+b01)
    #   M7 = (a01-a11)(b10+b11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6

    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    def strassen_block(a_idx, b_idx, w_idx, c0):
        """Add 7 rank-1 terms for one 2x2*2x2 Strassen block at column offset c0."""
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        w00, w01, w10, w11 = w_idx

        # M1 = (a00+a11)(b00+b11) -> contributes to C00(+), C11(+)
        U[a00][c0] += 1; U[a11][c0] += 1
        V[b00][c0] += 1; V[b11][c0] += 1
        W[w00][c0] += 1; W[w11][c0] += 1

        # M2 = (a10+a11)b00 -> contributes to C10(+), C11(-)
        U[a10][c0+1] += 1; U[a11][c0+1] += 1
        V[b00][c0+1] += 1
        W[w10][c0+1] += 1; W[w11][c0+1] -= 1

        # M3 = a00(b01-b11) -> contributes to C01(+), C11(+)
        U[a00][c0+2] += 1
        V[b01][c0+2] += 1; V[b11][c0+2] -= 1
        W[w01][c0+2] += 1; W[w11][c0+2] += 1

        # M4 = a11(b10-b00) -> contributes to C00(+), C10(+)
        U[a11][c0+3] += 1
        V[b10][c0+3] += 1; V[b00][c0+3] -= 1
        W[w00][c0+3] += 1; W[w10][c0+3] += 1

        # M5 = (a00+a01)b11 -> contributes to C00(-), C01(+)
        U[a00][c0+4] += 1; U[a01][c0+4] += 1
        V[b11][c0+4] += 1
        W[w00][c0+4] -= 1; W[w01][c0+4] += 1

        # M6 = (a10-a00)(b00+b01) -> contributes to C11(+)
        U[a10][c0+5] += 1; U[a00][c0+5] -= 1
        V[b00][c0+5] += 1; V[b01][c0+5] += 1
        W[w11][c0+5] += 1

        # M7 = (a01-a11)(b10+b11) -> contributes to C00(+)
        U[a01][c0+6] += 1; U[a11][c0+6] -= 1
        V[b10][c0+6] += 1; V[b11][c0+6] += 1
        W[w00][c0+6] += 1

    # A indices: A[i,j] -> U index i*m+j = i*4+j
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5  (left block)
    # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7  (right block)
    a_L = [0, 1, 4, 5]   # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]

    # B indices: B[j,k] -> V index j*p+k = j*6+k
    # B col block 0 (cols 0,1):
    #   rows 0,1: B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    #   rows 2,3: B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]

    # B col block 1 (cols 2,3):
    #   rows 0,1: B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9
    #   rows 2,3: B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]

    # B col block 2 (cols 4,5):
    #   rows 0,1: B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11
    #   rows 2,3: B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]

    # W indices: W[k*n+i]
    # Col block 0: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    w_1 = [0, 2, 1, 3]
    # Col block 1: C[0,2]->W[4], C[0,3]->W[6], C[1,2]->W[5], C[1,3]->W[7]
    w_2 = [4, 6, 5, 7]
    # Col block 2: C[0,4]->W[8], C[0,5]->W[10], C[1,4]->W[9], C[1,5]->W[11]
    w_3 = [8, 10, 9, 11]

    # 6 blocks: (a_part, b_rows, b_cols)
    # Block 1: a_L, b_1_T, w_1, col 0
    strassen_block(a_L, b_1_T, w_1, 0)
    # Block 2: a_R, b_1_B, w_1, col 7
    strassen_block(a_R, b_1_B, w_1, 7)
    # Block 3: a_L, b_2_T, w_2, col 14
    strassen_block(a_L, b_2_T, w_2, 14)
    # Block 4: a_R, b_2_B, w_2, col 21
    strassen_block(a_R, b_2_B, w_2, 21)
    # Block 5: a_L, b_3_T, w_3, col 28
    strassen_block(a_L, b_3_T, w_3, 28)
    # Block 6: a_R, b_3_B, w_3, col 35
    strassen_block(a_R, b_3_B, w_3, 35)

    # Verify the decomposition
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_np[i*m+j][r] * V_np[j*p+k][r] * W_np[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
