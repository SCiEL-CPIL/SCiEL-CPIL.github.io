import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def decompose(n, m, p):
    # Base cases
    if n == 1 and m == 1 and p == 1:
        # Rank 1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W

    if n == 1:
        # Dot product of length m for each k
        # C[k] = sum_j A[j] B[j,k]
        # Rank m*p
        rank = m * p
        U = [[0] * rank for _ in range(m)] # n*m = m
        V = [[0] * rank for _ in range(m * p)] # m*p
        W = [[0] * rank for _ in range(p)] # p*n = p
        
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W

    if p == 1:
        # Dot product of length m for each i
        # C[i] = sum_j A[i,j] B[j]
        # Rank n*m
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    if m == 1:
        # Outer product
        # C[i,k] = A[i] B[k]
        # Rank n*p
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W

    # Recursive case
    # Strategy: Split the largest dimension, or follow a specific pattern
    # To ensure we hit the Strassen base case (2,2,2), we should split
    # even dimensions into halves if possible.
    
    # If n, m, p are all even, we could use Strassen directly?
    # But our current logic splits into two independent problems.
    # The Strassen benefit only comes from the (2,2,2) base case in this simple splitter.
    # Let's try to split the dimension that allows reaching (2,2,2) or smaller.
    
    # Heuristic: Split the largest dimension. If tie, split n, then m, then p.
    # Actually, for (3,4,7), splitting n=3 is good.
    # For (2,4,7), splitting m=4 is good.
    # For (2,2,7), splitting p=7 is good.
    # For (2,2,4), splitting p=4 is good.
    # For (2,2,3), splitting p=3 is good.
    
    # Let's just split the largest dimension into two roughly equal parts.
    max_dim = max(n, m, p)
    
    if n == max_dim and n > 1:
        n1 = n // 2
        n2 = n - n1
        # Subproblem 1: n1 x m x p
        U1, V1, W1 = decompose(n1, m, p)
        # Subproblem 2: n2 x m x p
        U2, V2, W2 = decompose(n2, m, p)
        
        # Combine
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Map U1
        for i in range(n1):
            for j in range(m):
                for c in range(len(U1[0])):
                    U[i * m + j][c] = U1[i * m + j][c]
        # Map V1
        for j in range(m):
            for k in range(p):
                for c in range(len(V1[0])):
                    V[j * p + k][c] = V1[j * p + k][c]
        # Map W1
        for k in range(p):
            for i in range(n1):
                for c in range(len(W1[0])):
                    W[k * n + i][c] = W1[k * n1 + i][c]
        
        # Map U2
        offset = len(U1[0])
        for i in range(n2):
            for j in range(m):
                for c in range(len(U2[0])):
                    U[(i + n1) * m + j][offset + c] = U2[i * m + j][c]
        # Map V2
        for j in range(m):
            for k in range(p):
                for c in range(len(V2[0])):
                    V[j * p + k][offset + c] = V2[j * p + k][c]
        # Map W2
        for k in range(p):
            for i in range(n2):
                for c in range(len(W2[0])):
                    W[k * n + (i + n1)][offset + c] = W2[k * n2 + i][c]
                    
        return U, V, W

    elif m == max_dim and m > 1:
        m1 = m // 2
        m2 = m - m1
        # Subproblem 1: n x m1 x p
        U1, V1, W1 = decompose(n, m1, p)
        # Subproblem 2: n x m2 x p
        U2, V2, W2 = decompose(n, m2, p)
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Map U1
        for i in range(n):
            for j in range(m1):
                for c in range(len(U1[0])):
                    U[i * m + j][c] = U1[i * m1 + j][c]
        # Map V1
        for j in range(m1):
            for k in range(p):
                for c in range(len(V1[0])):
                    V[j * p + k][c] = V1[j * p + k][c]
        # Map W1
        for k in range(p):
            for i in range(n):
                for c in range(len(W1[0])):
                    W[k * n + i][c] = W1[k * n + i][c]
        
        # Map U2
        offset = len(U1[0])
        for i in range(n):
            for j in range(m2):
                for c in range(len(U2[0])):
                    U[i * m + (j + m1)][offset + c] = U2[i * m2 + j][c]
        # Map V2
        for j in range(m2):
            for k in range(p):
                for c in range(len(V2[0])):
                    V[(j + m1) * p + k][offset + c] = V2[j * p + k][c]
        # Map W2
        for k in range(p):
            for i in range(n):
                for c in range(len(W2[0])):
                    W[k * n + i][offset + c] = W2[k * n + i][c]
                    
        return U, V, W

    else: # p is max
        p1 = p // 2
        p2 = p - p1
        # Subproblem 1: n x m x p1
        U1, V1, W1 = decompose(n, m, p1)
        # Subproblem 2: n x m x p2
        U2, V2, W2 = decompose(n, m, p2)
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Map U1
        for i in range(n):
            for j in range(m):
                for c in range(len(U1[0])):
                    U[i * m + j][c] = U1[i * m + j][c]
        # Map V1
        for j in range(m):
            for k in range(p1):
                for c in range(len(V1[0])):
                    V[j * p + k][c] = V1[j * p1 + k][c]
        # Map W1
        for k in range(p1):
            for i in range(n):
                for c in range(len(W1[0])):
                    W[k * n + i][c] = W1[k * n + i][c]
        
        # Map U2
        offset = len(U1[0])
        for i in range(n):
            for j in range(m):
                for c in range(len(U2[0])):
                    U[i * m + j][offset + c] = U2[i * m + j][c]
        # Map V2
        for j in range(m):
            for k in range(p2):
                for c in range(len(V2[0])):
                    V[j * p + (k + p1)][offset + c] = V2[j * p2 + k][c]
        # Map W2
        for k in range(p2):
            for i in range(n):
                for c in range(len(W2[0])):
                    W[(k + p1) * n + i][offset + c] = W2[k * n + i][c]
                    
        return U, V, W

# Special case for Strassen (2,2,2)
def decompose_strassen_222():
    # Returns U, V, W for 2x2 * 2x2
    # U is 4x7, V is 4x7, W is 4x7
    # Indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 idx 0, A22 idx 3. B11 idx 0, B22 idx 3.
    # C11 += M1, C22 += M1
    # U[0][0]=1, U[3][0]=1
    # V[0][0]=1, V[3][0]=1
    # W[0][0]=1 (C11), W[3][0]=1 (C22)
    
    # M2 = (A21+A22)B11
    # A21 idx 2, A22 idx 3. B11 idx 0.
    # C21 += M2
    # U[2][1]=1, U[3][1]=1
    # V[0][1]=1
    # W[2][1]=1 (C21)
    
    # M3 = A11(B12-B22)
    # A11 idx 0. B12 idx 1, B22 idx 3.
    # C12 += M3
    # U[0][2]=1
    # V[1][2]=1, V[3][2]=-1
    # W[1][2]=1 (C12)
    
    # M4 = A22(B21-B11)
    # A22 idx 3. B21 idx 2, B11 idx 0.
    # C21 += M4
    # U[3][3]=1
    # V[2][3]=1, V[0][3]=-1
    # W[2][3]=1 (C21)
    
    # M5 = (A11+A12)B22
    # A11 idx 0, A12 idx 1. B22 idx 3.
    # C12 += M5, C22 += M5? No.
    # Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify the standard Strassen terms and assignments.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Assignments:
    # M1:
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2:
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3:
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4:
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # M5:
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1 # Wait, C22 uses M5?
    # Formula: C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
    # So M5 only contributes to C12?
    # Let's check C12 = M3 + M5.
    # M3 = A11(B12-B22). M5 = (A11+A12)B22 = A11 B22 + A12 B22.
    # M3 + M5 = A11 B12 - A11 B22 + A11 B22 + A12 B22 = A11 B12 + A12 B22. Correct.
    # So W[1][4] = 1. W[3][4] should be 0.
    W[3][4] = 0
    
    # M6:
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C22 += M6
    
    # M7:
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C11 += M7
    
    return U, V, W

# Override decompose to use Strassen for 2,2,2
_orig_decompose = decompose
def decompose(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return decompose_strassen_222()
    return _orig_decompose(n, m, p)
