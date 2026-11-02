import time

# Deadline guard: we must finish well before 150s wall clock.
_DEADLINE = time.time() + 145.0

DIMENSIONS = [2, 4, 4]


def _build_strassen_2x2x2():
    """
    Return the Strassen decomposition for M_{2,2,2} as (U, V, W) lists of shape (4,7) each.
    U indices: 0=A11, 1=A12, 2=A21, 3=A22
    V indices: 0=B11, 1=B12, 2=B21, 3=B22
    W indices: 0=C11, 1=C12, 2=C21, 3=C22
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]

    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1

    # M2 = (A21+A22)B11 -> contributes to C21, -C22
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1

    # M3 = A11(B12-B22) -> contributes to C12, C22
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1

    # M4 = A22(B21-B11) -> contributes to C11, C21
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[2][3] += 1

    # M5 = (A11+A12)B22 -> contributes to -C11, C12
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1

    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1

    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1

    return U, V, W


def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses 4 independent Strassen 2x2*2x2 blocks.
    
    Returns: {'U': list[8][28], 'V': list[16][28], 'W': list[8][28]}
    Indexing:
      U[i*4+j, r]  <-> A[i, j]
      V[i*4+j, r]  <-> B[i, j]
      W[k*2+i, r]  <-> C[i, k]   (official trace convention)
    """
    n, m, p = 2, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28

    sU, sV, sW = _build_strassen_2x2x2()

    def place_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, r0):
        """Add Strassen block for A[a_r0:a_r0+2, a_c0:a_c0+2] * B[b_r0:b_r0+2, b_c0:b_c0+2] -> C[c_r0:c_r0+2, c_c0:c_c0+2]"""
        for i in range(4):
            for r in range(7):
                if sU[i][r] != 0:
                    U[a_r0 * m + a_c0 + (i % 2)][r0 + r] += sU[i][r]
                if sV[i][r] != 0:
                    V[b_r0 * p + b_c0 + (i % 2)][r0 + r] += sV[i][r]
                if sW[i][r] != 0:
                    W[(c_c0 + (i % 2)) * n + c_r0 + (i // 2)][r0 + r] += sW[i][r]

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    place_block(0, 0, 0, 0, 0, 0, 0)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    place_block(0, 2, 2, 0, 0, 0, 7)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    place_block(0, 0, 0, 2, 0, 2, 14)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    place_block(0, 2, 2, 2, 0, 2, 21)

    return {"U": U, "V": V, "W": W}
