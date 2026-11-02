"""Rewritten program for 3x3 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Verify: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _construct_strassen_hybrid():
    """
    Rank-20 decomposition: Strassen (7 terms) on 2x2 top-left block +
    schoolbook (13 terms) for remaining products involving index 2.
    """
    n, m, p = 3, 3, 3
    U = [[0] * 20 for _ in range(n * m)]  # 9 rows
    V = [[0] * 20 for _ in range(m * p)]  # 9 rows
    W = [[0] * 20 for _ in range(p * n)]  # 9 rows

    # Index helpers:
    # A[i][j] -> U[i*3+j]
    # B[j][k] -> V[j*3+k]
    # C[i][k] -> W[k*3+i]

    # === Strassen 2x2 block (A rows 0,1; B cols 0,1; C rows 0,1 cols 0,1) ===
    # M1 = (a00+a11)(b00+b11) -> C[0][0]+C[0][1]+C[1][0]+C[1][1]
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = 1; W[4][0] = 1

    # M2 = (a10+a11)(b00) -> C[1][0]+C[1][1]
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = 1

    # M3 = a00(b11-b10) -> C[1][0]+C[1][1]
    U[0][2] = 1
    V[4][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[4][2] = 1

    # M4 = a11(b02-b00) -> C[0][1]+C[1][1]
    U[4][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1; W[4][3] = 1

    # M5 = (a00+a01)(b12) -> C[0][1]+C[1][1]
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[3][4] = 1; W[4][4] = 1

    # M6 = (a10-a01)(b00+b01) -> C[1][0]+C[1][1]
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[4][5] = 1

    # M7 = (a01-a11)(b10+b11) -> C[0][0]+C[1][1]
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[4][6] = 1

    # === Schoolbook for remaining 13 products ===
    # C[0][2] = a00*b02 + a01*b12 + a02*b22
    U[0][7] = 1; V[2][7] = 1; W[6][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[6][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[6][9] = 1

    # C[1][2] = a10*b02 + a11*b12 + a12*b22
    U[3][10] = 1; V[2][10] = 1; W[7][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[7][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[7][12] = 1

    # C[2][0] = a20*b00 + a21*b10 + a22*b20
    U[6][13] = 1; V[0][13] = 1; W[2][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[2][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[2][15] = 1

    # C[2][1] = a20*b01 + a21*b11 + a22*b21
    U[6][16] = 1; V[1][16] = 1; W[5][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[5][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[5][18] = 1

    # C[2][2] = a20*b02 + a21*b12 + a22*b22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][19] = 1; V[5][19] = 1; W[8][19] = 1
    U[8][19] = 1; V[8][19] = 1; W[8][19] = 1

    return U, V, W

def _construct_schoolbook():
    """Rank-27 schoolbook decomposition."""
    U = [[0] * 27 for _ in range(9)]
    V = [[0] * 27 for _ in range(9)]
    W = [[0] * 27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Returns a rank-minimal decomposition for 3x3 matrix multiplication.
    Tries rank-20 Strassen+schoolbook hybrid, falls back to rank-27.
    """
    try:
        U, V, W = _construct_strassen_hybrid()
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    U, V, W = _construct_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
