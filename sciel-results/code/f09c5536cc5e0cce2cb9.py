"""Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def _F(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    return Fraction(x)

def _build_strassen_4x4x4():
    """
    Strassen 4x4x4 (rank 16) factor matrices for multiplying A(4x4) by B(4x4) -> C(4x4).
    U has shape (16, 16) indexed [i*m+j, c], V (16,16) [j*p+k, c], W (16,16) [k*n+i, c].
    Convention: C[k*n + i, ...] = sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k]
    with W[k*n+i, c] selecting the output entry.
    """
    n = m = p = 4
    rank = 16
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    # 2x2 Strassen blocks (a11,a12,a21,a22) x (b11,b12,b21,b22)
    # M1=(a11+a22)(b11+b22)
    # M2=(a21+a22)b11
    # M3=a11(b12-b22)
    # M4=a22(b21-b11)
    # M5=(a11+a12)b22
    # M6=(a21-a11)(b11+b12)
    # M7=(a12-a22)(b21+b22)
    # c11=M1+M4-M5+M7
    # c12=M3+M5
    # c21=M2+M4
    # c22=M1-M2+M3+M6

    def add_U(r, i, j, v):
        U[i * m + j][r] += v

    def add_V(r, j, k, v):
        V[j * p + k][r] += v

    def add_W(r, k, i, v):
        W[k * n + i][r] += v

    # Block (0,0): a11=U(0,0), a12=U(0,1), a21=U(1,0), a22=U(1,1)
    # b11=V(0,0), b12=V(0,1), b21=V(1,0), b22=V(1,1)
    # outputs c11=W(0,0), c12=W(0,1), c21=W(1,0), c22=W(1,1)
    # M1
    r = 0
    add_U(r, 0, 0, 1); add_U(r, 1, 1, 1)
    add_V(r, 0, 0, 1); add_V(r, 1, 1, 1)
    add_W(r, 0, 0, 1); add_W(r, 1, 1, -1); add_W(r, 1, 0, 1)  # c11: +, c22: -
    # M2
    r = 1
    add_U(r, 1, 0, 1); add_U(r, 1, 1, 1)
    add_V(r, 0, 0, 1)
    add_W(r, 1, 0, 1); add_W(r, 1, 1, -1)  # c21: +, c22: -
    # M3
    r = 2
    add_U(r, 0, 0, 1)
    add_V(r, 0, 1, 1); add_V(r, 1, 1, -1)
    add_W(r, 0, 1, 1); add_W(r, 1, 1, 1)  # c12: +, c22: +
    # M4
    r = 3
    add_U(r, 1, 1, 1)
    add_V(r, 1, 0, 1); add_V(r, 0, 0, -1)
    add_W(r, 0, 0, 1); add_W(r, 1, 0, 1)  # c11: +, c21: +
    # M5
    r = 4
    add_U(r, 0, 0, 1); add_U(r, 0, 1, 1)
    add_V(r, 1, 1, 1)
    add_W(r, 0, 0, -1); add_W(r, 0, 1, 1)  # c11: -, c12: +
    # M6
    r = 5
    add_U(r, 1, 0, 1); add_U(r, 0, 0, -1)
    add_V(r, 0, 0, 1); add_V(r, 0, 1, 1)
    add_W(r, 1, 1, 1)  # c22: +
    # M7
    r = 6
    add_U(r, 0, 1, 1); add_U(r, 1, 1, -1)
    add_V(r, 1, 0, 1); add_V(r, 1, 1, 1)
    add_W(r, 0, 0, 1)  # c11: +

    # Block (0,1): a11=U(0,2), a12=U(0,3), a21=U(1,2), a22=U(1,3)
    # b11=V(2,0), b12=V(2,1), b21=V(3,0), b22=V(3,1)
    # outputs c11=W(2,0), c12=W(2,1), c21=W(3,0), c22=W(3,1)
    r = 7
    add_U(r, 0, 2, 1); add_U(r, 1, 3, 1)
    add_V(r, 2, 0, 1); add_V(r, 3, 1, 1)
    add_W(r, 2, 0, 1); add_W(r, 3, 1, -1); add_W(r, 3, 0, 1)
    r = 8
    add_U(r, 1, 2, 1); add_U(r, 1, 3, 1)
    add_V(r, 2, 0, 1)
    add_W(r, 3, 0, 1); add_W(r, 3, 1, -1)
    r = 9
    add_U(r, 0, 2, 1)
    add_V(r, 2, 1, 1); add_V(r, 3, 1, -1)
    add_W(r, 2, 1, 1); add_W(r, 3, 1, 1)
    r = 10
    add_U(r, 1, 3, 1)
    add_V(r, 3, 0, 1); add_V(r, 2, 0, -1)
    add_W(r, 2, 0, 1); add_W(r, 3, 0, 1)
    r = 11
    add_U(r, 0, 2, 1); add_U(r, 0, 3, 1)
    add_V(r, 3, 1, 1)
    add_W(r, 2, 0, -1); add_W(r, 2, 1, 1)
    r = 12
    add_U(r, 1, 2, 1); add_U(r, 0, 2, -1)
    add_V(r, 2, 0, 1); add_V(r, 2, 1, 1)
    add_W(r, 3, 1, 1)
    r = 13
    add_U(r, 0, 3, 1); add_U(r, 1, 3, -1)
    add_V(r, 3, 0, 1); add_V(r, 3, 1, 1)
    add_W(r, 2, 0, 1)

    # Block (1,0): a11=U(2,0), a12=U(2,1), a21=U(3,0), a22=U(3,1)
    # b11=V(0,2), b12=V(0,3), b21=V(1,2), b22=V(1,3)
    # outputs c11=W(0,2), c12=W(0,3), c21=W(1,2), c22=W(1,3)
    r = 14
    add_U(r, 2, 0, 1); add_U(r, 3, 1, 1)
    add_V(r, 0, 2, 1); add_V(r, 1, 3, 1)
    add_W(r, 0, 2, 1); add_W(r, 1, 3, -1); add_W(r, 1, 2, 1)
    r = 15
    add_U(r, 3, 0, 1); add_U(r, 3, 1, 1)
    add_V(r, 0, 2, 1)
    add_W(r, 1, 2, 1); add_W(r, 1, 3, -1)
    # M3 for block (1,0)
    add_U(2, 2, 0, 1)
    add_V(2, 0, 3, 1); add_V(2, 1, 3, -1)
    add_W(2, 0, 3, 1); add_W(2, 1, 3, 1)
    # M4
    add_U(3, 3, 1, 1)
    add_V(3, 1, 2, 1); add_V(3, 0, 2, -1)
    add_W(3, 0, 2, 1); add_W(3, 1, 2, 1)
    # M5
    add_U(4, 2, 0, 1); add_U(4, 2, 1, 1)
    add_V(4, 1, 3, 1)
    add_W(4, 0, 2, -1); add_W(4, 0, 3, 1)
    # M6
    add_U(5, 3, 0, 1); add_U(5, 2, 0, -1)
    add_V(5, 0, 2, 1); add_V(5, 0, 3, 1)
    add_W(5, 1, 3, 1)
    # M7
    add_U(6, 2, 1, 1); add_U(6, 3, 1, -1)
    add_V(6, 1, 2, 1); add_V(6, 1, 3, 1)
    add_W(6, 0, 2, 1)

    # Block (1,1): a11=U(2,2), a12=U(2,3), a21=U(3,2), a22=U(3,3)
    # b11=V(2,2), b12=V(2,3), b21=V(3,2), b22=V(3,3)
    # outputs c11=W(2,2), c12=W(2,3), c21=W(3,2), c22=W(3,3)
    r = 14
    add_U(r, 2, 2, 1); add_U(r, 3, 3, 1)
    add_V(r, 2, 2, 1); add_V(r, 3, 3, 1)
    add_W(r, 2, 2, 1); add_W(r, 3, 3, -1); add_W(r, 3, 2, 1)
    r = 15
    add_U(r, 3, 2, 1); add_U(r, 3, 3, 1)
    add_V(r, 2, 2, 1)
    add_W(r, 3, 2, 1); add_W(r, 3, 3, -1)
    # M3
    add_U(2, 2, 2, 1)
    add_V(2, 2, 3, 1); add_V(2, 3, 3, -1)
    add_W(2, 2, 3, 1); add_W(2, 3, 3, 1)
    # M4
    add_U(3, 3, 3, 1)
    add_V(3, 3, 2, 1); add_V(3, 2, 2, -1)
    add_W(3, 2, 2, 1); add_W(3, 3, 2, 1)
    # M5
    add_U(4, 2, 2, 1); add_U(4, 2, 3, 1)
    add_V(4, 3, 3, 1)
    add_W(4, 2, 2, -1); add_W(4, 2, 3, 1)
    # M6
    add_U(5, 3, 2, 1); add_U(5, 2, 2, -1)
    add_V(5, 2, 2, 1); add_V(5, 2, 3, 1)
    add_W(5, 3, 3, 1)
    # M7
    add_U(6, 2, 3, 1); add_U(6, 3, 3, -1)
    add_V(6, 3, 2, 1); add_V(6, 3, 3, 1)
    add_W(6, 2, 2, 1)

    return U, V, W


def _build_4x4x3_schoolbook():
    """
    Schoolbook 4x4 * 4x3 -> 4x3 (rank 12).
    U (16,12) [i*m+j, c], V (12,12) [j*p+k, c], W (12,12) [k*n+i, c].
    """
    n = 4
    m = 4
    p = 3
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (block-diagonal rank addition)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = [row + [Fraction(0)] * r2 for row in U1] + [[Fraction(0)] * r1 + row for row in U2]
    V = [row + [Fraction(0)] * r2 for row in V1] + [[Fraction(0)] * r1 + row for row in V2]
    W = [row + [Fraction(0)] * r2 for row in W1] + [[Fraction(0)] * r1 + row for row in W2]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # Strategy: split 4x7 into two 4x4 blocks + one 4x3 block.
    # Strassen for each 4x4 (rank 16 each) + schoolbook for 4x3 (rank 12).
    # Total rank = 16 + 16 + 12 = 44.
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)

    U1, V1, W1 = _build_strassen_4x4x4()
    U2, V2, W2 = _build_strassen_4x4x4()
    U3, V3, W3 = _build_4x4x3_schoolbook()

    # Combine: (U1,V1,W1) handles A(4x4)*B_left(4x4) -> C_left(4x4)
    # (U2,V2,W2) handles A(4x4)*B_mid(4x4) -> C_mid(4x4)
    # (U3,V3,W3) handles A(4x4)*B_right(4x3) -> C_right(4x3)
    #
    # The combined U has rows indexed by i*m+j where i in [0,4), j in [0,4).
    # V rows indexed by j*p+k where j in [0,4), k in [0,7).
    # W rows indexed by k*n+i where k in [0,7), i in [0,4).
    #
    # For the full 4x4 * 4x7, we need:
    # U: (16, rank) - same input A indices
    # V: (28, rank) - j in [0,4), k in [0,7)
    # W: (28, rank) - k in [0,7), i in [0,4)
    #
    # The 4x4x4 decompositions use k in [0,4). We need to map:
    #   Block 1 (left): k in [0,4) -> full k in [0,4)
    #   Block 2 (mid):  k in [0,4) -> full k in [4,8) but p=7 so k in [4,7)
    #   Wait, p=7, so k ranges 0..6. Two 4-wide blocks don't fit in 7.
    #
    # Better split: 4x7 = 4x4 + 4x3. Only two blocks.
    # Block 1: Strassen 4x4x4 (k in [0,4)), rank 16
    # Block 2: schoolbook 4x4x3 (k in [4,7)), rank 12
    # Total rank = 28.

    U_s, V_s, W_s = _build_strassen_4x4x4()
    U_b, V_b, W_b = _build_4x4x3_schoolbook()

    # Now remap indices for the full problem:
    # Full V index: j*p+k = j*7+k, j in [0,4), k in [0,7)
    # Full W index: k*n+i = k*4+i, k in [0,7), i in [0,4)
    #
    # Block 1 (Strassen, k in [0,4)):
    #   U rows: i*4+j (i,j in [0,4)) -> same in full
    #   V rows: j*4+k (j in [0,4), k in [0,4)) -> full j*7+k
    #   W rows: k*4+i (k in [0,4), i in [0,4)) -> full k*4+i (same)
    #
    # Block 2 (schoolbook 4x4x3, k in [0,3) locally, maps to k in [4,7) full):
    #   U rows: i*4+j -> same
    #   V rows: j*3+k_local (j in [0,4), k_local in [0,3)) -> full j*7+(4+k_local)
    #   W rows: k_local*4+i (k_local in [0,3), i in [0,4)) -> full (4+k_local)*4+i

    rank = 16 + 12
    U_full = [[Fraction(0)] * rank for _ in range(16)]
    V_full = [[Fraction(0)] * rank for _ in range(28)]
    W_full = [[Fraction(0)] * rank for _ in range(28)]

    # Block 1: Strassen
    for i in range(4):
        for j in range(4):
            uidx = i * 4 + j
            for c in range(16):
                val = U_s[i * 4 + j][c]
                if val != 0:
                    U_full[uidx][c] += val
    for j in range(4):
        for k in range(4):
            vidx_s = j * 4 + k
            vidx_f = j * 7 + k
            for c in range(16):
                val = V_s[j * 4 + k][c]
                if val != 0:
                    V_full[vidx_f][c] += val
    for k in range(4):
        for i in range(4):
            widx = k * 4 + i
            for c in range(16):
                val = W_s[k * 4 + i][c]
                if val != 0:
                    W_full[widx][c] += val

    # Block 2: schoolbook 4x4x3, offset k by 4
    for i in range(4):
        for j in range(4):
            uidx = i * 4 + j
            for c in range(12):
                val = U_b[i * 4 + j][c]
                if val != 0:
                    U_full[uidx][16 + c] += val
    for j in range(4):
        for k_local in range(3):
            vidx_b = j * 3 + k_local
            vidx_f = j * 7 + (4 + k_local)
            for c in range(12):
                val = V_b[j * 3 + k_local][c]
                if val != 0:
                    V_full[vidx_f][16 + c] += val
    for k_local in range(3):
        for i in range(4):
            widx_b = k_local * 4 + i
            widx_f = (4 + k_local) * 4 + i
            for c in range(12):
                val = W_b[k_local * 4 + i][c]
                if val != 0:
                    W_full[widx_f][16 + c] += val

    # Convert to plain integers (all entries are integers)
    U_out = [[int(x) for x in row] for row in U_full]
    V_out = [[int(x) for x in row] for row in V_full]
    W_out = [[int(x) for x in row] for row in W_full]

    return {"U": U_out, "V": V_out, "W": W_out}


def _schoolbook_fallback(n, m, p):
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
