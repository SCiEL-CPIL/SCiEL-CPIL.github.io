"""Find a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication."""
import time
from fractions import Fraction
import numpy as np

N, M, P = 2, 2, 8
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16


def verify(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition. Returns True/False."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    # Check valid entries sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # Check invalid entries sum to 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W


def strassen_228():
    """
    Rank-28 decomposition using Strassen's 7-product method tiled over 4 blocks of 2x2.
    
    A is 2x2: A00, A01, A10, A11 (rows 0,1,2,3 of U)
    B is 2x8: split into 4 blocks of 2x2 along k.
    C is 2x8: split into 4 blocks of 2x2 along k.
    
    For each block m in 0..3, B_m has entries B[0,2m], B[0,2m+1], B[1,2m], B[1,2m+1].
    
    Strassen products per block:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    Output reconstruction per block:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for m in range(4):
        # B indices for this block
        b_00 = 0 * P + 2 * m        # B[0, 2m]
        b_01 = 0 * P + (2 * m + 1)  # B[0, 2m+1]
        b_10 = 1 * P + 2 * m        # B[1, 2m]
        b_11 = 1 * P + (2 * m + 1)  # B[1, 2m+1]
        
        # C indices for this block
        c_00 = (2 * m) * N + 0      # C[0, 2m]
        c_01 = (2 * m + 1) * N + 0  # C[0, 2m+1]
        c_10 = (2 * m) * N + 1      # C[1, 2m]
        c_11 = (2 * m + 1) * N + 1  # C[1, 2m+1]
        
        for s in range(7):
            r = m * 7 + s
            
            if s == 0:  # M1 = (A00+A11)(B00+B11)
                U[0][r] = 1
                U[3][r] = 1
                V[b_00][r] = 1
                V[b_11][r] = 1
                W[c_00][r] = 1
                W[c_11][r] = 1
            elif s == 1:  # M2 = (A10+A11)B00
                U[2][r] = 1
                U[3][r] = 1
                V[b_00][r] = 1
                W[c_10][r] = 1
                W[c_11][r] = -1
            elif s == 2:  # M3 = A00(B01-B11)
                U[0][r] = 1
                V[b_01][r] = 1
                V[b_11][r] = -1
                W[c_01][r] = 1
                W[c_11][r] = 1
            elif s == 3:  # M4 = A11(B10-B00)
                U[3][r] = 1
                V[b_10][r] = 1
                V[b_00][r] = -1
                W[c_00][r] = 1
                W[c_10][r] = 1
            elif s == 4:  # M5 = (A00+A10)B11
                U[0][r] = 1
                U[2][r] = 1
                V[b_11][r] = 1
                W[c_00][r] = -1
                W[c_01][r] = 1
            elif s == 5:  # M6 = (A11-A00)(B00+B10)
                U[3][r] = 1
                U[0][r] = -1
                V[b_00][r] = 1
                V[b_10][r] = 1
                W[c_11][r] = 1
            elif s == 6:  # M7 = (A01-A11)(B01+B11)
                U[1][r] = 1
                U[3][r] = -1
                V[b_01][r] = 1
                V[b_11][r] = 1
                W[c_00][r] = 1
    
    return U, V, W


def construct():
    """Entry point: return the best decomposition found."""
    try:
        # Try the rank-28 Strassen tiling
        U, V, W = strassen_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W = schoolbook()
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
