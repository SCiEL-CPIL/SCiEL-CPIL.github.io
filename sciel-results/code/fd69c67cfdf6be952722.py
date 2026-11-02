import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Cache for decompositions
    cache = {}
    
    def get_decomp(n, m, p):
        key = (n, m, p)
        if key in cache:
            return cache[key]
        
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            cache[key] = (U, V, W)
            return U, V, W
            
        # Base case: 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # U: 4x7, V: 4x7, W: 4x7
            # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = 1
            # M2
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = -1
            # M3
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[3][2] = 1
            # M4
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            # M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[1][4] = -1; W[0][4] = -1
            # M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            # M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[1][6] = 1
            
            cache[key] = (U, V, W)
            return U, V, W

        # Recursive case
        # Try splitting m
        if m % 2 == 0:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            # Combine
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Fill U1, V1, W1
            for i in range(n):
                for j in range(m1):
                    for c in range(len(U1[0])):
                        U[i*m + j][c] = U1[i*m1 + j][c]
            
            for j in range(m1):
                for k in range(p):
                    for c in range(len(V1[0])):
                        V[j*p + k][c] = V1[j*p + k][c]
            
            for k in range(p):
                for i in range(n):
                    for c in range(len(W1[0])):
                        W[k*n + i][c] = W1[k*n + i][c]
                        
            # Fill U2, V2, W2
            offset = len(U1[0])
            for i in range(n):
                for j in range(m2):
                    for c in range(len(U2[0])):
                        U[i*m + (m1 + j)][offset + c] = U2[i*m2 + j][c]
            
            for j in range(m2):
                for k in range(p):
                    for c in range(len(V2[0])):
                        V[(m1 + j)*p + k][offset + c] = V2[j*p + k][c]
            
            for k in range(p):
                for i in range(n):
                    for c in range(len(W2[0])):
                        W[k*n + i][offset + c] = W2[k*n + i][c]
                        
            cache[key] = (U, V, W)
            return U, V, W

        # Try splitting p
        if p % 2 == 0:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(len(U1[0])):
                        U[i*m + j][c] = U1[i*m + j][c]
                        U[i*m + j][len(U1[0]) + c] = U2[i*m + j][c]
                        
            for j in range(m):
                for k in range(p1):
                    for c in range(len(V1[0])):
                        V[j*p + k][c] = V1[j*p1 + k][c]
                for k in range(p2):
                    for c in range(len(V2[0])):
                        V[j*p + (p1 + k)][len(V1[0]) + c] = V2[j*p2 + k][c]
                        
            for k in range(p1):
                for i in range(n):
                    for c in range(len(W1[0])):
                        W[k*n + i][c] = W1[k*n + i][c]
            for k in range(p2):
                for i in range(n):
                    for c in range(len(W2[0])):
                        W[(p1 + k)*n + i][len(W1[0]) + c] = W2[k*n + i][c]
                        
            cache[key] = (U, V, W)
            return U, V, W

        # Try splitting n
        if n % 2 == 0:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            for i in range(n1):
                for j in range(m):
                    for c in range(len(U1[0])):
                        U[i*m + j][c] = U1[i*m + j][c]
            for i in range(n2):
                for j in range(m):
                    for c in range(len(U2[0])):
                        U[(n1 + i)*m + j][len(U1[0]) + c] = U2[i*m + j][c]
                        
            for j in range(m):
                for k in range(p):
                    for c in range(len(V1[0])):
                        V[j*p + k][c] = V1[j*p + k][c]
                        V[j*p + k][len(V1[0]) + c] = V2[j*p + k][c]
                        
            for k in range(p):
                for i in range(n1):
                    for c in range(len(W1[0])):
                        W[k*n + i][c] = W1[k*n + i][c]
                for i in range(n2):
                    for c in range(len(W2[0])):
                        W[k*n + (n1 + i)][len(W1[0]) + c] = W2[k*n + i][c]
                        
            cache[key] = (U, V, W)
            return U, V, W

        # Fallback: Schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        cache[key] = (U, V, W)
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}

