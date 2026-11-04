import numpy as np
from fractions import Fraction
import sys
import time

DIMENSIONS = [3, 5, 7]
DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    # Output: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W index convention: W[k*n + i, r]
    
    rank_terms = []  # list of (U_vec, V_vec, W_vec), each of length nm, mp, pn respectively
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Strassen 2x2x2 decomposition: 7 terms
    # For A[2,2], B[2,2], C[2,2]:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10-B00)       -> C00 += M3
    # M4 = A11(B01-B11)       -> C11 += M4
    # M5 = (A00+A01)B11       -> C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C01 += M6, C00 -= M6
    # M7 = (A01-A11)(B10+B11) -> C10 += M7, C11 -= M7
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        
        def get_u(i, j): return idx_u(i0+i, j0+j)
        def get_v(j, k): return idx_v(j0+j, k0+k)
        def get_w(k, i): return idx_w(k0+k, i0+i)
        
        # M1 = (A00+A11)(B00+B11)
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(0,0)] = 1.0; Uv[get_u(1,1)] = 1.0
        Vv[get_v(0,0)] = 1.0; Vv[get_v(1,1)] = 1.0
        Wv[get_w(0,0)] = 1.0; Wv[get_w(1,1)] = 1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M2 = (A10+A11)B00
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(1,0)] = 1.0; Uv[get_u(1,1)] = 1.0
        Vv[get_v(0,0)] = 1.0
        Wv[get_w(0,1)] = 1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M3 = A00(B10-B00)
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(0,0)] = 1.0
        Vv[get_v(1,0)] = 1.0; Vv[get_v(0,0)] = -1.0
        Wv[get_w(0,0)] = 1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M4 = A11(B01-B11)
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(1,1)] = 1.0
        Vv[get_v(0,1)] = 1.0; Vv[get_v(1,1)] = -1.0
        Wv[get_w(1,1)] = 1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M5 = (A00+A01)B11
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(0,0)] = 1.0; Uv[get_u(0,1)] = 1.0
        Vv[get_v(1,1)] = 1.0
        Wv[get_w(1,0)] = 1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M6 = (A10-A00)(B00+B01)
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(1,0)] = 1.0; Uv[get_u(0,0)] = -1.0
        Vv[get_v(0,0)] = 1.0; Vv[get_v(0,1)] = 1.0
        Wv[get_w(1,0)] = 1.0; Wv[get_w(0,0)] = -1.0
        rank_terms.append((Uv, Vv, Wv))
        
        # M7 = (A01-A11)(B10+B11)
        Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
        Uv[get_u(0,1)] = 1.0; Uv[get_u(1,1)] = -1.0
        Vv[get_v(1,0)] = 1.0; Vv[get_v(1,1)] = 1.0
        Wv[get_w(0,1)] = 1.0; Wv[get_w(1,1)] = -1.0
        rank_terms.append((Uv, Vv, Wv))

    # Strategy: Use Strassen for i=0,1 blocks where possible
    # A is 3x5, B is 5x7, C is 3x7
    # i ranges 0,1,2; j ranges 0-4; k ranges 0-6
    
    # For i=0,1: We can use Strassen blocks
    # Tile j and k with 2x2 blocks
    # j: 0,1 | 2,3 | 4 (single)
    # k: 0,1 | 2,3 | 4,5 | 6 (single)
    
    # Strassen blocks for i=0,1:
    # (j=0,1, k=0,1): rank 7
    # (j=0,1, k=2,3): rank 7
    # (j=0,1, k=4,5): rank 7
    # (j=2,3, k=0,1): rank 7
    # (j=2,3, k=2,3): rank 7
    # (j=2,3, k=4,5): rank 7
    # (j=0,1, k=6): schoolbook, 2 terms (i=0,1)
    # (j=2,3, k=6): schoolbook, 2 terms (i=0,1)
    # (j=4, k=0,1): schoolbook, 2 terms
    # (j=4, k=2,3): schoolbook, 2 terms
    # (j=4, k=4,5): schoolbook, 2 terms
    # (j=4, k=6): schoolbook, 1 term
    
    # For i=2: schoolbook for all j,k
    # j=0..4, k=0..6: 5*7 = 35 terms
    
    # Total for i=0,1: 6*7 + 2+2+2+2+2+1 = 42 + 11 = 53
    # Total for i=2: 35
    # Grand total: 53 + 35 = 88
    
    # Add Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            if j0 + 1 < m and k0 + 1 < p:
                add_strassen_block(0, j0, k0)
    
    # Schoolbook for remaining i=0,1 cases
    # j=0,1 with k=6
    for i in [0, 1]:
        for j in [0, 1]:
            Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
            Uv[idx_u(i, j)] = 1.0
            Vv[idx_v(j, 6)] = 1.0
            Wv[idx_w(6, i)] = 1.0
            rank_terms.append((Uv, Vv, Wv))
    
    # j=2,3 with k=6
    for i in [0, 1]:
        for j in [2, 3]:
            Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
            Uv[idx_u(i, j)] = 1.0
            Vv[idx_v(j, 6)] = 1.0
            Wv[idx_w(6, i)] = 1.0
            rank_terms.append((Uv, Vv, Wv))
    
    # j=4 with k=0,1,2,3,4,5,6
    for i in [0, 1]:
        for k in range(7):
            Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
            Uv[idx_u(i, 4)] = 1.0
            Vv[idx_v(4, k)] = 1.0
            Wv[idx_w(k, i)] = 1.0
            rank_terms.append((Uv, Vv, Wv))
    
    # Schoolbook for i=2
    for j in range(5):
        for k in range(7):
            Uv = [0.0] * (n*m); Vv = [0.0] * (m*p); Wv = [0.0] * (p*n)
            Uv[idx_u(2, j)] = 1.0
            Vv[idx_v(j, k)] = 1.0
            Wv[idx_w(k, 2)] = 1.0
            rank_terms.append((Uv, Vv, Wv))
    
    # Build output matrices
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for r, (Uv, Vv, Wv) in enumerate(rank_terms):
        for idx in range(n*m):
            if Uv[idx] != 0:
                U[idx][r] = Uv[idx]
        for idx in range(m*p):
            if Vv[idx] != 0:
                V[idx][r] = Vv[idx]
        for idx in range(p*n):
            if Wv[idx] != 0:
                W[idx][r] = Wv[idx]
    
    return {"U": U, "V": V, "W": W}
