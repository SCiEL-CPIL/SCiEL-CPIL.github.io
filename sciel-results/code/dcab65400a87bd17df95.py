import numpy as np
from functools import lru_cache

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
@lru_cache(maxsize=None)
def get_rank(n, m, p):
    # Base case: Schoolbook
    if n == 1 or m == 1 or p == 1:
        return n * m * p
    
    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        return 7

    best_rank = n * m * p
    best_split = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = get_rank(n1, m, p)
        r2 = get_rank(n2, m, p)
        # If we split n, V is shared? No, the rank is sum of ranks.
        # The V matrices can be the same size, but the rank is additive.
        total = r1 + r2
        if total < best_rank:
            best_rank = total
            best_split = ('n', n1, n2)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = get_rank(n, m1, p)
        r2 = get_rank(n, m2, p)
        total = r1 + r2
        if total < best_rank:
            best_rank = total
            best_split = ('m', m1, m2)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = get_rank(n, m, p1)
        r2 = get_rank(n, m, p2)
        total = r1 + r2
        if total < best_rank:
            best_rank = total
            best_split = ('p', p1, p2)
            
    return best_rank

def construct_decomp(n, m, p):
    # Returns U, V, W as lists of lists
    rank = get_rank(n, m, p)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base case: Schoolbook
    if n == 1 or m == 1 or p == 1:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W
        # U indices: i*m+j. i,j in 0,1.
        # V indices: j*p+k. j,k in 0,1.
        # W indices: k*n+i. k,i in 0,1.
        
        # Term 1:
        # A: 00, 11. B: 00, 11. C: 00, 01, 10, 11.
        U[0*2+0][0] = 1; U[1*2+1][0] = 1
        V[0*2+0][0] = 1; V[1*2+1][0] = 1
        W[0*2+0][0] = 1; W[0*2+1][0] = 1; W[1*2+0][0] = 1; W[1*2+1][0] = 1
        
        # Term 2:
        # A: 10, 11. B: 00. C: 10, 11.
        U[1*2+0][1] = 1; U[1*2+1][1] = 1
        V[0*2+0][1] = 1
        W[0*2+1][1] = 1; W[1*2+1][1] = 1
        
        # Term 3:
        # A: 00. B: 01, 11. C: 01, 11.
        U[0*2+0][2] = 1
        V[0*2+1][2] = 1; V[1*2+1][2] = -1
        W[1*2+0][2] = 1; W[1*2+1][2] = 1
        
        # Term 4:
        # A: 11. B: 10, 00. C: 10, 11.
        U[1*2+1][3] = 1
        V[1*2+0][3] = 1; V[0*2+0][3] = -1
        W[0*2+1][3] = 1; W[1*2+1][3] = 1
        
        # Term 5:
        # A: 00, 01. B: 11. C: 01, 11.
        U[0*2+0][4] = 1; U[0*2+1][4] = 1
        V[1*2+1][4] = 1
        W[1*2+0][4] = 1; W[1*2+1][4] = 1
        
        # Term 6:
        # A: 10, 00. B: 00, 01. C: 10, 11.
        U[1*2+0][5] = 1; U[0*2+0][5] = -1
        V[0*2+0][5] = 1; V[0*2+1][5] = 1
        W[0*2+1][5] = 1; W[1*2+1][5] = 1
        
        # Term 7:
        # A: 01, 11. B: 10, 11. C: 10, 11.
        U[0*2+1][6] = 1; U[1*2+1][6] = -1
        V[1*2+0][6] = 1; V[1*2+1][6] = 1
        W[0*2+1][6] = 1; W[1*2+1][6] = 1
        
        return U, V, W

    # Recursive case
    # Find best split
    best_rank = n * m * p
    best_split = None
    
    for n1 in range(1, n):
        n2 = n - n1
        total = get_rank(n1, m, p) + get_rank(n2, m, p)
        if total < best_rank:
            best_rank = total
            best_split = ('n', n1, n2)
            
    for m1 in range(1, m):
        m2 = m - m1
        total = get_rank(n, m1, p) + get_rank(n, m2, p)
        if total < best_rank:
            best_rank = total
            best_split = ('m', m1, m2)
            
    for p1 in range(1, p):
        p2 = p - p1
        total = get_rank(n, m, p1) + get_rank(n, m, p2)
        if total < best_rank:
            best_rank = total
            best_split = ('p', p1, p2)
            
    if not best_split:
        # Fallback to schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    axis, d1, d2 = best_split
    
    if axis == 'n':
        U1, V1, W1 = construct_decomp(d1, m, p)
        U2, V2, W2 = construct_decomp(d2, m, p)
        r1 = len(V1[0])
        r2 = len(V2[0])
        
        # Combine U
        # U1 is d1*m x r1, U2 is d2*m x r2
        # U is n*m x (r1+r2)
        for i in range(d1 * m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(d2 * m):
            for c in range(r2):
                U[d1 * m + i][r1 + c] = U2[i][c]
                
        # Combine V
        # V is m*p x (r1+r2)
        # V1 and V2 are the same? No, they are decompositions for the same B.
        # But the rank is additive, so we just put them side by side.
        for j in range(m * p):
            for c in range(r1):
                V[j][c] = V1[j][c]
            for c in range(r2):
                V[j][r1 + c] = V2[j][c]
                
        # Combine W
        # W is p*n x (r1+r2)
        for k in range(p * d1):
            for c in range(r1):
                W[k][c] = W1[k][c]
        for k in range(p * d2):
            for c in range(r2):
                W[p * d1 + k][r1 + c] = W2[k][c]
                
    elif axis == 'm':
        U1, V1, W1 = construct_decomp(n, d1, p)
        U2, V2, W2 = construct_decomp(n, d2, p)
        r1 = len(V1[0])
        r2 = len(V2[0])
        
        # Combine U
        for i in range(n * d1):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(n * d2):
            for c in range(r2):
                U[n * d1 + i][r1 + c] = U2[i][c]
                
        # Combine V
        for j in range(d1 * p):
            for c in range(r1):
                V[j][c] = V1[j][c]
        for j in range(d2 * p):
            for c in range(r2):
                V[d1 * p + j][r1 + c] = V2[j][c]
                
        # Combine W
        for k in range(p * n):
            for c in range(r1):
                W[k][c] = W1[k][c]
            for c in range(r2):
                W[k][r1 + c] = W2[k][c]
                
    elif axis == 'p':
        U1, V1, W1 = construct_decomp(n, m, d1)
        U2, V2, W2 = construct_decomp(n, m, d2)
        r1 = len(V1[0])
        r2 = len(V2[0])
        
        # Combine U
        for i in range(n * m):
            for c in range(r1):
                U[i][c] = U1[i][c]
            for c in range(r2):
                U[i][r1 + c] = U2[i][c]
                
        # Combine V
        for j in range(m * d1):
            for c in range(r1):
                V[j][c] = V1[j][c]
        for j in range(m * d2):
            for c in range(r2):
                V[m * d1 + j][r1 + c] = V2[j][c]
                
        # Combine W
        for k in range(d1 * n):
            for c in range(r1):
                W[k][c] = W1[k][c]
        for k in range(d2 * n):
            for c in range(r2):
                W[d1 * n + k][r1 + c] = W2[k][c]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = construct_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
