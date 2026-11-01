"""
Rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Score = seed / rank = 16 / 14 = 8/7 ≈ 1.1429 (optimal rank for this tensor).
"""
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 140.0  # safety margin under 150 s wall-clock

# EVOLVE-BLOCK-START

def _make_rank14():
    """
    Build the rank-14 decomposition using two Strassen 2x2x2 blocks.

    Index conventions:
      A is 2x2:  U row a = i*2+j  (i = output row, j = input col of A / row of B)
      B is 2x4:  V row b = j*4+k  (j = row of B, k = col of B)
      C is 2x4:  W row c = k*2+i  (k = col of C, i = row of C)

    Left block (B columns 0,1) and Right block (B columns 2,3) each
    implement a standard Strassen 2x2x2 multiplication with 7 multiplications.
    """
    R = 14
    U = [[0] * R for _ in range(4)]
    V = [[0] * R for _ in range(8)]
    W = [[0] * R for _ in range(8)]

    # --- Left Strassen block (output columns 0, 1) ---
    # m1 = (A00+A11)(B00+B11)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1

    # m2 = (A10+A11) B00
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1

    # m3 = A00 (B01-B11)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1

    # m4 = A11 (B10-B00)
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1

    # m5 = (A00+A01) B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1

    # m6 = (A10-A00)(B00+B01)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1

    # m7 = (A01-A11)(B10+B11)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1

    # --- Right Strassen block (output columns 2, 3) ---
    # m1 = (A00+A11)(B02+B13)
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1

    # m2 = (A10+A11) B02
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1

    # m3 = A00 (B03-B13)
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1

    # m4 = A11 (B12-B02)
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1

    # m5 = (A00+A01) B13
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1

    # m6 = (A10-A00)(B02+B03)
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1

    # m7 = (A01-A11)(B12+B13)
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1

    return U, V, W


def _check_decomposition(U, V, W):
    """
    Verify that the factorisation is exact:
      sum_r U[a,r] V[b,r] W[c,r]  ==  1  when  a//2 == c%2  and  a%2 == b//4
      sum_r U[a,r] V[b,r] W[c,r]  ==  0  otherwise
    Uses einsum for speed; tensor size is 4*8*8 = 256 scalars.
    """
    UA = np.asarray(U, dtype=np.float64)
    VA = np.asarray(V, dtype=np.float64)
    WA = np.asarray(W, dtype=np.float64)
    T = np.einsum('ar,br,cr->abc', UA, VA, WA)

    # Valid triples: a=i*2+j, b=j*4+k, c=k*2+i  =>  i=a//2, j=a%2, j=b//4, k=b%4, i=c%2
    for a in range(4):
        i = a >> 1
        j = a & 1
        for b in range(8):
            jb = b >> 2
            k = b & 3
            if j == jb:
                c = k * 2 + i
                if abs(T[a, b, c] - 1.0) > 1e-12:
                    return False
            # invalid c values for this (a,b): the other c
            c_other = k * 2 + (1 - i)
            if c_other != c:
                if abs(T[a, b, c_other]) > 1e-12:
                    return False
    # Also check triples where j != jb (all c are invalid)
    for a in range(4):
        j = a & 1
        for b in range(8):
            jb = b >> 2
            if j != jb:
                for c in range(8):
                    if abs(T[a, b, c]) > 1e-12:
                        return False
    return True


def _schoolbook_rank16():
    """Trivial rank-16 (schoolbook) fallback – should never be returned if rank-14 is correct."""
    R = 16
    U = [[0] * R for _ in range(4)]
    V = [[0] * R for _ in range(8)]
    W = [[0] * R for _ in range(8)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * 2 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    return U, V, W


def construct():
    if time.time() - _T0 > _DEADLINE:
        U, V, W = _make_rank14()
        return {'U': U, 'V': V, 'W': W}

    U, V, W = _make_rank14()

    if _check_decomposition(U, V, W):
        return {'U': U, 'V': V, 'W': W}

    # Fallback (should not trigger)
    U2, V2, W2 = _schoolbook_rank16()
    return {'U': U2, 'V': V2, 'W': W2}

# EVOLVE-BLOCK-END
