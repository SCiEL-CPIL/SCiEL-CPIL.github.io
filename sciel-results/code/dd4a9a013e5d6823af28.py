"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1; W[3][1]=-1
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1; W[3][2]=1
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return U, V, W

def _build_strassen_4x4x2():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    def add_block(ai, aj, bj, ci, ck, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    c = U2[li*2+lj][t]
                    if c:
                        U[(ai+li)*4 + (aj+lj)][col] += c
            for lj in range(2):
                for lk in range(2):
                    c = V2[lj*2+lk][t]
                    if c:
                        V[(bj+lj)*2 + (ck+lk)][col] += c
            for lk in range(2):
                for li in range(2):
                    c = W2[lk*2+li][t]
                    if c:
                        W[(ck+lk)*4 + (ci+li)][col] += c
    
    add_block(0,0, 0, 0,0, 0)
    add_block(0,2, 2, 0,0, 7)
    add_block(2,0, 0, 2,0, 14)
    add_block(2,2, 2, 2,0, 21)
    return U, V, W

def _build_4x5_5x6():
    n, m, p = 4, 5, 6
    U1, V1, W1 = _build_strassen_4x4()
    U2, V2, W2 = _build_schoolbook(4, 1, 6)
    U3, V3, W3 = _build_strassen_4x4x2()
    
    total_rank = 49 + 24 + 28 # 101
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # 1. 4x4x4 (rows 0-3 of A, cols 0-3 of B, cols 0-3 of C)
    r = 0
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*5+j][r+c] = U1[i*4+j][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*6+k][r+c] = V1[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][r+c] = W1[k*4+i][c]
    r = 49
    
    # 2. 4x1x6 (rows 0-3 of A, col 4 of B, all cols of C)
    for i in range(4):
        for c in range(24):
            U[i*5+4][r+c] = U2[i*1+0][c]
    for k in range(6):
        for c in range(24):
            V[4*6+k][r+c] = V2[0*6+k][c]
    for k in range(6):
        for i in range(4):
            for c in range(24):
                W[k*4+i][r+c] = W2[k*4+i][c]
    r = 49 + 24
    
    # 3. 4x4x2 (rows 0-3 of A, cols 0-3 of B, cols 4-5 of C)
    for i in range(4):
        for j in range(4):
            for c in range(28):
                U[i*5+j][r+c] = U3[i*4+j][c]
    for j in range(4):
        for k in range(2):
            for c in range(28):
                V[j*6+(k+4)][r+c] = V3[j*2+k][c]
    for k in range(2):
        for i in range(4):
            for c in range(28):
                W[(k+4)*4+i][r+c] = W3[k*4+i][c]
                
    return U, V, W

def construct():
    U, V, W = _build_4x5_5x6()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

