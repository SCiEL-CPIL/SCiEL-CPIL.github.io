import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 100  # safety margin
    
    # Try to build rank-105 decomposition
    # Part 1: 4x4x4 (Strassen, rank 49)
    # Part 2: 4x4x2 (schoolbook, rank 32)
    # Part 3: 4x1x6 (schoolbook, rank 24)
    
    # Define 2x2x2 Strassen terms
    # After careful derivation, the 7 terms are:
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    
    # Verify 2x2x2
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for Uc, Vc, Wc in strassen_2x2:
                    total += Uc[i*2+j] * Vc[j*2+k] * Wc[k*2+i]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: schoolbook for everything, rank 120
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
    
    # Build 4x4x4 via Kronecker product
    U4_terms = []
    V4_terms = []
    W4_terms = []
    for U2a, V2a, W2a in strassen_2x2:
        for U2b, V2b, W2b in strassen_2x2:
            U4 = [0]*16
            V4 = [0]*16
            W4 = [0]*16
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U4[i*4+j] = U2a[i1*2+j1] * U2b[i2*2+j2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V4[j*4+k] = V2a[j1*2+k1] * V2b[j2*2+k2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W4[k*4+i] = W2a[k1*2+i1] * W2b[k2*2+i2]
            U4_terms.append(U4)
            V4_terms.append(V4)
            W4_terms.append(W4)
    
    # Verify 4x4x4
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0
                for c in range(49):
                    total += U4_terms[c][i*4+j] * V4_terms[c][j*4+k] * W4_terms[c][k*4+i]
                if total != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fallback: schoolbook
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
    
    # Build full 4x5x6 decomposition, rank 105
    total_rank = 49 + 32 + 24
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    # Part 1: 4x4x4 Strassen
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i*m + j][col] = U4_terms[c][i*4+j]
        for j in range(4):
            for k in range(4):
                V[j*p + k][col] = V4_terms[c][j*4+k]
        for k in range(4):
            for i in range(4):
                W[k*n + i][col] = W4_terms[c][k*4+i]
        col += 1
    
    # Part 2: 4x4x2 schoolbook (j=0..3, k=4..5)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Part 3: 4x1x6 schoolbook (j=4, k=0..5)
    for i in range(4):
        for k in range(6):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
