import time
import sys
from functools import lru_cache

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 110.0  # 10s margin

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of matrix multiplication M_{4,5,6}.
    
    Uses recursive block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
        """Schoolbook decomposition for M_{n,m,p} with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W
    
    def strassen_2x2x2():
        """Strassen decomposition for M_{2,2,2} with rank 7."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (a00 + a11)(b00 + b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # m2 = (a10 + a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # m3 = a00(b01 - b11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # m4 = a11(b10 - b00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # m5 = (a00 + a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    memo = {}
    
    def get_decomp(n, m, p):
        if time.time() > _DEADLINE:
            return schoolbook(n, m, p)

        key = (n, m, p)
        if key in memo:
            return memo[key]
        
        # Base case: 2x2x2 uses Strassen
        if n == 2 and m == 2 and p == 2:
            res = strassen_2x2x2()
            memo[key] = res
            return res
        
        # For small cases, use schoolbook
        if n * m * p <= 8:
            res = schoolbook(n, m, p)
            memo[key] = res
            return res
        
        best = None
        best_rank = float('inf')
        
        # Split n
        if n >= 2:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                
                r1 = len(U1[0])
                r2 = len(U2[0])
                r = r1 + r2
                
                if r >= best_rank:
                    continue

                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # Fill U
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m + j][c] = U1[i*m + j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(n1+i)*m + j][r1+c] = U2[i*m + j][c]
                
                # Fill V
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p + k][c]
                        for c in range(r2):
                            V[j*p + k][r1+c] = V2[j*p + k][c]
                
                # Fill W
                for k in range(p):
                    for i in range(n1):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n1 + i][c]
                    for i in range(n2):
                        for c in range(r2):
                            W[k*n + (n1+i)][r1+c] = W2[k*n2 + i][c]
                
                best_rank = r
                best = (U, V, W)
        
        # Split m
        if m >= 2:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                
                r1 = len(U1[0])
                r2 = len(U2[0])
                r = r1 + r2
                
                if r >= best_rank:
                    continue

                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # Fill U
                for i in range(n):
                    for j in range(m1):
                        for c in range(r1):
                            U[i*m + j][c] = U1[i*m1 + j][c]
                    for j in range(m2):
                        for c in range(r2):
                            U[i*m + (m1+j)][r1+c] = U2[i*m2 + j][c]
                
                # Fill V
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p + k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(m1+j)*p + k][r1+c] = V2[j*p + k][c]
                
                # Fill W
                for k in range(p):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n + i][c]
                        for c in range(r2):
                            W[k*n + i][r1+c] = W2[k*n + i][c]
                
                best_rank = r
                best = (U, V, W)
        
        # Split p
        if p >= 2:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                
                r1 = len(U1[0])
                r2 = len(U2[0])
                r = r1 + r2
                
                if r >= best_rank:
                    continue

                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # Fill U
                for i in range(n*m):
                    for c in range(r1):
                        U[i][c] = U1[i][c]
                    for c in range(r2):
                        U[i][r1+c] = U2[i][c]
                
                # Fill V
                for j in range(m):
                    for k in range(p1):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p1 + k][c]
                    for k in range(p2):
                        for c in range(r2):
                            V[j*p + (p1+k)][r1+c] = V2[j*p2 + k][c]
                
                # Fill W
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n + i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(p1+k)*n + i][r1+c] = W2[k*n + i][c]
                
                best_rank = r
                best = (U, V, W)
        
        if best is not None:
            memo[key] = best
            return best
        else:
            res = schoolbook(n, m, p)
            memo[key] = res
            return res
    
    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
