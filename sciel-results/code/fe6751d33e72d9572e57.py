import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = build_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def build_decomposition(n, m, p):
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 1 and m == 1:
        # C[0,0] = A[0,0] * sum B[0,k] ? No.
        # C is 1xP. C[0,k] = A[0,0] B[0,k].
        # Rank P.
        U = [[1]*p]
        V = [[0]*p for _ in range(p)]
        W = [[0]*p for _ in range(p)]
        for k in range(p):
            V[0][k] = 1 # B[0,k] index 0*p+k = k
            W[k][k] = 1 # C[0,k] index k*1+0 = k
        return U, V, W

    if n == 1 and p == 1:
        # C[0,0] = sum A[0,j] B[j,0]
        # Rank M.
        U = [[0]*m for _ in range(1)] # Shape 1xM
        V = [[0]*m for _ in range(m)] # Shape Mx1
        W = [[0]*m for _ in range(1)] # Shape 1xM
        for j in range(m):
            U[0][j] = 1
            V[j][0] = 1
            W[0][j] = 1
        return U, V, W

    if m == 1 and p == 1:
        # C[i,0] = sum A[i,0] B[0,0] ? No.
        # C[i,0] = A[i,0] B[0,0].
        # Rank N.
        U = [[0]*n for _ in range(n)] # Shape NxN
        V = [[0]*n for _ in range(n)] # Shape Nx1
        W = [[0]*n for _ in range(n)] # Shape NxN
        for i in range(n):
            U[i][i] = 1
            V[i][0] = 1
            W[i][i] = 1
        return U, V, W

    if n == 2 and m == 2 and p == 2:
        return strassen_2x2x2()

    # Recursive cases
    # Try to split p
    if p >= 2:
        p1 = 2 if p >= 2 else 1
        p2 = p - p1
        if p1 > 0 and p2 > 0:
            U1, V1, W1 = build_decomposition(n, m, p1)
            U2, V2, W2 = build_decomposition(n, m, p2)
            return concatenate(U1, U2, n*m), concatenate(V1, V2, m*p1), concatenate(W1, W2, p1*n)

    # Try to split n
    if n >= 2:
        n1 = 2 if n >= 2 else 1
        n2 = n - n1
        if n1 > 0 and n2 > 0:
            U1, V1, W1 = build_decomposition(n1, m, p)
            U2, V2, W2 = build_decomposition(n2, m, p)
            return concatenate(U1, U2, n1*m), concatenate(V1, V2, m*p), concatenate(W1, W2, p*n1)

    # Try to split m
    if m >= 2:
        m1 = 2 if m >= 2 else 1
        m2 = m - m1
        if m1 > 0 and m2 > 0:
            U1, V1, W1 = build_decomposition(n, m1, p)
            U2, V2, W2 = build_decomposition(n, m2, p)
            return concatenate(U1, U2, n*m1), concatenate(V1, V2, m1*p), concatenate(W1, W2, p*n)

    # Fallback to schoolbook
    return schoolbook(n, m, p)

def concatenate(U1, U2, rows):
    # U1 is rows x r1, U2 is rows x r2
    # Result is rows x (r1+r2)
    r1 = len(U1[0])
    r2 = len(U2[0])
    U = [[0]*(r1+r2) for _ in range(rows)]
    for i in range(rows):
        for j in range(r1):
            U[i][j] = U1[i][j]
        for j in range(r2):
            U[i][r1+j] = U2[i][j]
    return U

def schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def strassen_2x2x2():
    # Returns U, V, W for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # P3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # P4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1
    
    # P5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[3][4] = -1; W[1][4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W
