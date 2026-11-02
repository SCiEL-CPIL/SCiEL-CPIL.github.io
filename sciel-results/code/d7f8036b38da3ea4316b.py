import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _build_target(n, m, p):
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

def _verify(U, V, W, tgt, n, m, p):
    R = len(U[0])
    acc = {}
    for r in range(R):
        u_nz = [(a, U[a][r]) for a in range(len(U)) if U[a][r] != 0]
        v_nz = [(b, V[b][r]) for b in range(len(V)) if V[b][r] != 0]
        w_nz = [(c, W[c][r]) for c in range(len(W)) if W[c][r] != 0]
        for a, ua in u_nz:
            for b, vb in v_nz:
                uv = ua * vb
                for c, wc in w_nz:
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _add_strassen_2x2x2(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    # a_base: index of A[0,0]
    # b_base: index of B[0,0]
    # c_base: index of C[0,0]
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: a_base, a_base+1, a_base+m, a_base+m+1
    # B indices: b_base, b_base+1, b_base+p, b_base+p+1
    # C indices: c_base, c_base+n, c_base+n+1, c_base+2n+1 ? No.
    # C is p x n.
    # C[0,0] is c_base.
    # C[0,1] is c_base+1.
    # C[1,0] is c_base+n.
    # C[1,1] is c_base+n+1.
    
    A00 = a_base
    A01 = a_base + 1
    A10 = a_base + m
    A11 = a_base + m + 1
    
    B00 = b_base
    B01 = b_base + 1
    B10 = b_base + p
    B11 = b_base + p + 1
    
    C00 = c_base
    C01 = c_base + 1
    C10 = c_base + n
    C11 = c_base + n + 1
    
    # M1
    U[A00][rank_offset] += 1
    U[A11][rank_offset] += 1
    V[B00][rank_offset] += 1
    V[B11][rank_offset] += 1
    W[C00][rank_offset] += 1
    W[C11][rank_offset] += 1
    
    # M2
    U[A10][rank_offset+1] += 1
    U[A11][rank_offset+1] += 1
    V[B00][rank_offset+1] += 1
    W[C10][rank_offset+1] += 1
    
    # M3
    U[A00][rank_offset+2] += 1
    V[B10][rank_offset+2] += 1
    V[B11][rank_offset+2] -= 1
    W[C01][rank_offset+2] += 1
    
    # M4
    U[A11][rank_offset+3] += 1
    V[B01][rank_offset+3] += 1
    V[B00][rank_offset+3] -= 1
    W[C00][rank_offset+3] += 1
    W[C10][rank_offset+3] += 1
    
    # M5
    U[A00][rank_offset+4] += 1
    U[A01][rank_offset+4] += 1
    V[B11][rank_offset+4] += 1
    W[C01][rank_offset+4] += 1
    W[C00][rank_offset+4] -= 1
    
    # M6
    U[A10][rank_offset+5] += 1
    U[A00][rank_offset+5] -= 1
    V[B00][rank_offset+5] += 1
    V[B01][rank_offset+5] += 1
    W[C11][rank_offset+5] += 1
    
    # M7
    U[A01][rank_offset+6] += 1
    U[A11][rank_offset+6] -= 1
    V[B10][rank_offset+6] += 1
    V[B11][rank_offset+6] += 1
    W[C00][rank_offset+6] += 1
    
    return rank_offset + 7

def _add_schoolbook_2x2x1(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    # A 2x2, B 2x1, C 2x1
    # B indices: b_base, b_base+p
    # C indices: c_base, c_base+n
    
    A00 = a_base
    A01 = a_base + 1
    A10 = a_base + m
    A11 = a_base + m + 1
    
    B00 = b_base
    B10 = b_base + p
    
    C00 = c_base
    C10 = c_base + n
    
    U[A00][rank_offset] += 1
    V[B00][rank_offset] += 1
    W[C00][rank_offset] += 1
    
    U[A10][rank_offset+1] += 1
    V[B00][rank_offset+1] += 1
    W[C10][rank_offset+1] += 1
    
    U[A01][rank_offset+2] += 1
    V[B10][rank_offset+2] += 1
    W[C00][rank_offset+2] += 1
    
    U[A11][rank_offset+3] += 1
    V[B10][rank_offset+3] += 1
    W[C10][rank_offset+3] += 1
    
    return rank_offset + 4

def _add_strassen_2x2x3(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    # A 2x2, B 2x3, C 2x3
    # B cols 0,1,2
    # C cols 0,1,2
    
    # Strassen for cols 0,1
    r = _add_strassen_2x2x2(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p)
    # Schoolbook for col 2
    r = _add_schoolbook_2x2x1(U, V, W, a_base, b_base + 2, c_base + 2, r, n, m, p)
    
    return r

def _add_schoolbook_1x2x6(U, V, W, a_base, b_base, c_base, rank_offset, n, m, p):
    # A 1x2, B 2x6, C 1x6
    # A indices: a_base, a_base+1
    # B indices: b_base..b_base+5, b_base+p..b_base+p+5
    # C indices: c_base..c_base+5
    
    A0 = a_base
    A1 = a_base + 1
    
    for j in range(6):
        B0j = b_base + j
        B1j = b_base + p + j
        Cj = c_base + j
        
        U[A0][rank_offset] += 1
        V[B0j][rank_offset] += 1
        W[Cj][rank_offset] += 1
        rank_offset += 1
        
        U[A1][rank_offset] += 1
        V[B1j][rank_offset] += 1
        W[Cj][rank_offset] += 1
        rank_offset += 1
        
    return rank_offset

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    # n=3, m=4, p=6
    # U shape (12, rank), V shape (24, rank), W shape (18, rank)
    
    # Estimate rank: 68
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # A11 B_top_L
    # A11 base: 0
    # B_top_L base: 0
    # C base: 0
    r = _add_strassen_2x2x3(U, V, W, 0, 0, 0, r, n, m, p)
    
    # A11 B_top_R
    # A11 base: 0
    # B_top_R base: 3
    # C base: 3
    r = _add_strassen_2x2x3(U, V, W, 0, 3, 3, r, n, m, p)
    
    # A12 B_bot_L
    # A12 base: 2
    # B_bot_L base: 12
    # C base: 0
    r = _add_strassen_2x2x3(U, V, W, 2, 12, 0, r, n, m, p)
    
    # A12 B_bot_R
    # A12 base: 2
    # B_bot_R base: 15
    # C base: 3
    r = _add_strassen_2x2x3(U, V, W, 2, 15, 3, r, n, m, p)
    
    # A31 B_top
    # A31 base: 8
    # B_top base: 0
    # C base: 6 (Row 2 of C starts at 6*3+2? No. C is 6x3. Row 2 is indices 6,7,8? No.
    # C[i, j] index is i*3 + j.
    # Row 0: 0,1,2
    # Row 1: 3,4,5
    # Row 2: 6,7,8
    # So C base for row 2 is 6.
    r = _add_schoolbook_1x2x6(U, V, W, 8, 0, 6, r, n, m, p)
    
    # A32 B_bot
    # A32 base: 10
    # B_bot base: 12
    # C base: 6
    r = _add_schoolbook_1x2x6(U, V, W, 10, 12, 6, r, n, m, p)
    
    if r != rank:
        # If rank estimation was wrong, we might have issues.
        # But we can just return what we have.
        pass

    tgt = _build_target(n, m, p)
    if _verify(U, V, W, tgt, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
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
