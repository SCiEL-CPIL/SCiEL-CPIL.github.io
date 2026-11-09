import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def standard_term(U, V, W, i, j, k, col):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        return col + 1
    
    def strassen_2x2x2(U, V, W, i1, i2, j1, j2, k1, k2, col):
        # M1 = (a11+a22)(b11+b22)
        U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M2 = (a21+a22)b11
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
        col += 1
        # M3 = a11(b12-b22)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M4 = a22(b21-b11)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
        col += 1
        # M5 = (a11+a12)b22
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
        col += 1
        # M6 = (a21-a11)(b11+b12)
        U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        # M7 = (a12-a22)(b21+b22)
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        return col
    
    def standard_2x2x1(U, V, W, i1, i2, j1, j2, k, col):
        # C[i1,k] = A[i1,j1]*B[j1,k] + A[i1,j2]*B[j2,k]
        # C[i2,k] = A[i2,j1]*B[j1,k] + A[i2,j2]*B[j2,k]
        col = standard_term(U, V, W, i1, j1, k, col)
        col = standard_term(U, V, W, i1, j2, k, col)
        col = standard_term(U, V, W, i2, j1, k, col)
        col = standard_term(U, V, W, i2, j2, k, col)
        return col
    
    def standard_2x1x2(U, V, W, i1, i2, j, k1, k2, col):
        # C[i1,k1] = A[i1,j]*B[j,k1]
        # C[i1,k2] = A[i1,j]*B[j,k2]
        # C[i2,k1] = A[i2,j]*B[j,k1]
        # C[i2,k2] = A[i2,j]*B[j,k2]
        col = standard_term(U, V, W, i1, j, k1, col)
        col = standard_term(U, V, W, i1, j, k2, col)
        col = standard_term(U, V, W, i2, j, k1, col)
        col = standard_term(U, V, W, i2, j, k2, col)
        return col
    
    def standard_2x1x1(U, V, W, i1, i2, j, k, col):
        col = standard_term(U, V, W, i1, j, k, col)
        col = standard_term(U, V, W, i2, j, k, col)
        return col
    
    # Partition: i: {0,1},{2,3}; j: {0,1},{2,3},{4}; k: {0,1},{2,3},{4}
    i_groups = [(0,1), (2,3)]
    j_groups = [(0,1), (2,3), (4,4)]
    k_groups = [(0,1), (2,3), (4,4)]
    
    # Count expected rank
    expected_rank = 0
    for (i1,i2) in i_groups:
        for (j1,j2) in j_groups:
            for (k1,k2) in k_groups:
                ni = i2 - i1 + 1
                nj = j2 - j1 + 1
                nk = k2 - k1 + 1
                if ni == 2 and nj == 2 and nk == 2:
                    expected_rank += 7
                elif ni == 2 and nj == 2 and nk == 1:
                    expected_rank += 4
                elif ni == 2 and nj == 1 and nk == 2:
                    expected_rank += 4
                elif ni == 2 and nj == 1 and nk == 1:
                    expected_rank += 2
                else:
                    expected_rank += ni * nj * nk
    
    rank = expected_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for (i1, i2) in i_groups:
        for (j1, j2) in j_groups:
            for (k1, k2) in k_groups:
                ni = i2 - i1 + 1
                nj = j2 - j1 + 1
                nk = k2 - k1 + 1
                
                if ni == 2 and nj == 2 and nk == 2:
                    col = strassen_2x2x2(U, V, W, i1, i2, j1, j2, k1, k2, col)
                elif ni == 2 and nj == 2 and nk == 1:
                    col = standard_2x2x1(U, V, W, i1, i2, j1, j2, k1, col)
                elif ni == 2 and nj == 1 and nk == 2:
                    col = standard_2x1x2(U, V, W, i1, i2, j1, k1, k2, col)
                elif ni == 2 and nj == 1 and nk == 1:
                    col = standard_2x1x1(U, V, W, i1, i2, j1, k1, col)
                else:
                    for i in range(i1, i2+1):
                        for j in range(j1, j2+1):
                            for k in range(k1, k2+1):
                                col = standard_term(U, V, W, i, j, k, col)
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = standard_term(U, V, W, i, j, k, col)
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
